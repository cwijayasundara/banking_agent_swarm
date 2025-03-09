import os
from retriever import get_query_engine
from llama_index.core.agent.workflow import AgentWorkflow
from llama_index.llms.gemini import Gemini
from datetime import datetime
from llama_index.core.workflow import Context
from customer_db import create_banking_customer_db
import pandas as pd
from dotenv import load_dotenv
from pending_tx_agent import create_pending_tx_query_engine
from llama_index.core.tools import FunctionTool
from llama_index.core.workflow import (
    step,
    Event,
    Context,
    StartEvent,
    StopEvent,
    Workflow,
)
from llama_index.core.agent import FunctionCallingAgent

today = datetime.now().strftime("%d/%m/%Y")

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

query_engine = get_query_engine()

customer_db_query_engine = create_banking_customer_db()

pending_tx_query_engine = create_pending_tx_query_engine()

llm = Gemini(model="models/gemini-1.5-pro",api_key=GOOGLE_API_KEY)

async def search_interest_rates(question: str) -> str:
    """Ask a question to the bank account interest rate documents stored in the vector index."""
    response = query_engine.query(question)
    return str(response)

async def search_customer_details(question: str) -> str:
    """Ask a question to the bank customer database which contains customer and account information in a SQL database."""
    response = customer_db_query_engine.query(question)
    return str(response)

async def search_pending_tx_details(question: str) -> str:
    """Get the total amount of pending transactions for a customer from a Pandas dataframe."""
    response = pending_tx_query_engine.query(question)
    return str(response)

# tools 
interest_rate_rag_tool = FunctionTool.from_defaults(
    fn=search_interest_rates,
    name="account_interest_rates",
    description="search the account interest rates from the bank account interest rate documents stored in the vector index",
)  

customer_details_tool = FunctionTool.from_defaults(
    fn=search_customer_details,
    name="customer_details",
    description="search the customer details from the bank customer database which contains customer and account information in a SQL database",
)

pending_tx_details_tool = FunctionTool.from_defaults(
    fn=search_pending_tx_details,
    name="pending_tx_details",
    description="search the total amount of pending transactions for a customer from a Pandas dataframe",
)

# events def
class OutlineEvent(Event):
    outline: str

class QuestionEvent(Event):
    question: str

class AnswerEvent(Event):
    question: str
    answer: str

class ReviewEvent(Event):
    final_answer: str

class ProgressEvent(Event):
    progress: str

class CustomerInvestmentAdvisorAgent(Workflow):

    @step()
    async def formulate_plan(
        self, ctx: Context, ev: StartEvent
    ) -> OutlineEvent:
        query = ev.query
        await ctx.set("original_query", query)
        await ctx.set("tools", ev.tools)

        prompt = f"""You are an expert in formulating plans to answer queries by customers on their bank account interest rates, customer details and pending transactions. : {query}     """
    
        response = await llm.acomplete(prompt)

        ctx.write_event_to_stream(
            ProgressEvent(progress="Outline:\n" + str(response))
        )

        return OutlineEvent(outline=str(response))
    
    # formulate some questions based on the outline
    @step()
    async def formulate_questions(
        self, ctx: Context, ev: OutlineEvent
    ) -> QuestionEvent:
        outline = ev.outline
        await ctx.set("outline", outline)

        prompt = f"""You are an expert in formulating questions to answer queries by customers on their bank account interest rates, customer details and pending transactions.
        If the query is about bank account interest rates then formulate questions to find the interest rates based on the outline: {outline}. Generate only 1 question.
        If the query is about customer details then formulate questions to find the customer details based on the outline: {outline}. Generate only 1 question.
        If the query is about pending transactions then formulate questions to find the pending transactions based on the outline: {outline}. Generate only 1 question.
        If the query is something complex pls use the below guidelines:
        - If the user asks how much money they can unvest in an ISA account then formulate questions as follows:
        - Whats the total amount they can invest in an ISA account? The infoamtion is stored in the vector index.
        - Whats their bank balance? The information is stored in the SQL database.
        - If the user asks about pending transactions then formulate questions as follows:
        - Whats the total amount of pending transactions? The information is stored in the Pandas dataframe : {outline}.
        Generate 1 question each for the above scenarios.
        """
        response = await llm.acomplete(prompt)

        questions = str(response).split("\n")
        questions = [x for x in questions if x]

        ctx.write_event_to_stream(
            ProgressEvent(
                progress="Formulated questions:\n" + "\n".join(questions)
            )
        )

        await ctx.set("num_questions", len(questions))

        ctx.write_event_to_stream(
            ProgressEvent(progress="Questions:\n" + "\n".join(questions))
        )

        for question in questions:
            ctx.send_event(QuestionEvent(question=question))
    

    @step()
    async def answer_question(
        self, ctx: Context, ev: QuestionEvent
    ) -> AnswerEvent:
        question = ev.question
        if (
            not question
            or question.isspace()
            or question == ""
            or question is None
        ):
            ctx.write_event_to_stream(
                ProgressEvent(progress=f"Skipping empty question.")
            )  # Log skipping empty question
            return None
        agent = FunctionCallingAgent.from_tools(
            await ctx.get("tools"),
            verbose=True,
        )
        response = await agent.aquery(question)

        ctx.write_event_to_stream(
            ProgressEvent(
                progress=f"To question '{question}' the agent answered: {response}"
            )
        )

        return AnswerEvent(question=question, answer=str(response))
    

    @step()
    async def write_final_answer(self, ctx: Context, ev: AnswerEvent) -> ReviewEvent:
        # wait until we receive as many answers as there are questions
        num_questions = await ctx.get("num_questions")
        results = ctx.collect_events(ev, [AnswerEvent] * num_questions)
        if results is None:
            return None

        # maintain a list of all questions and answers no matter how many times this step is called
        try:
            previous_questions = await ctx.get("previous_questions")
        except:
            previous_questions = []
        previous_questions.extend(results)
        await ctx.set("previous_questions", previous_questions)

        prompt = f"""You are an expert in answring customer queries on account interest rates, customer details and pending transactions. 
        The outline is in <outline> and the questions and answers are in <questions> and <answers>.

        <outline>{await ctx.get('outline')}</outline>"""

        for result in previous_questions:
            prompt += f"<question>{result.question}</question>\n<answer>{result.answer}</answer>\n"

        ctx.write_event_to_stream(
            ProgressEvent(progress="Writing answer with prompt:\n" + prompt)
        )

        answer = await llm.acomplete(prompt)

        return ReviewEvent(answer=str(answer))
    
    @step
    async def review_answer(
        self, ctx: Context, ev: ReviewEvent
    ) -> StopEvent | QuestionEvent:
        # we re-review a maximum of 3 times
        try:
            num_reviews = await ctx.get("num_reviews")
        except:
            num_reviews = 1
        num_reviews += 1
        await ctx.set("num_reviews", num_reviews)

        answer = ev.answer

        prompt = f"""You are an expert reviewer of answers to customer queries on account interest rates, customer details and pending transactions. You are given an original query,
        and an answer that was written to satisfy that query. Review the answer and determine
        if it adequately answers the query and contains enough detail. If it doesn't, come up with
        a set of questions that will get you the facts necessary to expand the answer. Another
        agent will answer those questions. Your response should just be a list of questions, one
        per line, without any preamble or explanation. For speed, generate a maximum of 4 questions.
        The original query is: '{await ctx.get('original_query')}'.
        The answer is: <answer>{answer}</answer>.
        If the answer is fine, return just the string 'OKAY'."""

        # response = await Settings.llm.acomplete(prompt)
        response = await llm.acomplete(prompt)

        if response == "OKAY" or await ctx.get("num_reviews") >= 3:
            ctx.write_event_to_stream(
                ProgressEvent(progress="Answer is fine")
            )
            return StopEvent(result=answer)
        else:
            questions = str(response).split("\n")
            await ctx.set("num_questions", len(questions))
            ctx.write_event_to_stream(
                ProgressEvent(progress="Formulated some more questions")
            )
            for question in questions:
                ctx.send_event(QuestionEvent(question=question))
