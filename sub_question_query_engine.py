import os
import json
import asyncio
import nest_asyncio
from dotenv import load_dotenv
from llama_index.core.workflow import (
    step,
    Event,
    Context,
    StartEvent,
    StopEvent,
    Workflow,
)
from llama_index.llms.gemini import Gemini
from llama_index.core.tools import FunctionTool
from retriever import get_query_engine
from customer_db import create_banking_customer_db
from pending_tx_agent import get_pending_tx_details
from llama_index.core.agent import ReActAgent
from llama_index.llms.openai import OpenAI
from wf_agents import interest_rate_rag_tool, customer_details_tool, pending_tx_details_tool

# Apply nest_asyncio to allow nested event loops
nest_asyncio.apply()

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

llm = OpenAI(model="gpt-4o-mini",api_key=OPENAI_API_KEY)

# llm = Gemini(model="models/gemini-2.0-flash-001",api_key=GOOGLE_API_KEY)

class QueryEvent(Event):
    question: str

class AnswerEvent(Event):
    question: str
    answer: str

class SubQuestionQueryEngine(Workflow):

    @step(pass_context=True)
    async def query(self, ctx: Context, ev: StartEvent) -> QueryEvent:
        """Generate sub-questions based on the original query."""
        # Store the original query in the context
        await ctx.set("original_query", ev.query)
        print(f"Query is {ev.query}")
        
        # Store the LLM in the context
        await ctx.set("llm", ev.llm)
        
        # Store the tools in the context
        await ctx.set("tools", ev.tools)
        
        # Get the LLM from the context
        llm = await ctx.get("llm")
        original_query = await ctx.get("original_query")
        tools = await ctx.get("tools")
        
        # Generate sub-questions using the LLM
        response = llm.complete(f"""
        You are a helpful assistant that breaks down complex questions into simpler sub-questions.
        
        For the given user question, generate a list of sub-questions that would help answer the original question.
        
        The sub-questions should be specific and answerable using the available tools.
        
        Here is the user question: {original_query}
        
        And here is the list of tools: {tools}
        
        Return your response as a JSON object with the following format:
        ```json
        {{
            "sub_questions": [
                "sub-question 1",
                "sub-question 2",
                ...
            ]
        }}
        ```
        
        Only return the JSON object, nothing else.
        """)
        
        print(f"Sub-questions are {response}")
        
        try:
            # Try to parse the JSON response
            response_obj = json.loads(str(response))
        except json.JSONDecodeError:
            # If JSON parsing fails, extract JSON from the response or create a default
            print("Failed to parse JSON response. Using default format.")
            # Try to extract JSON from the response using regex
            import re
            json_match = re.search(r'```json\s*(.*?)\s*```', str(response), re.DOTALL)
            
            if json_match:
                try:
                    response_obj = json.loads(json_match.group(1))
                except json.JSONDecodeError:
                    # If extraction fails, use the original query as a fallback
                    response_obj = {"sub_questions": [original_query]}
            else:
                # If no JSON block found, use the original query as a fallback
                response_obj = {"sub_questions": [original_query]}
        
        # Store the sub-questions in the context
        await ctx.set("sub_questions", response_obj["sub_questions"])
        await ctx.set("answers", [])
        
        # Return the first sub-question
        return QueryEvent(question=response_obj["sub_questions"][0])

    @step(pass_context=True)
    async def sub_question(self, ctx: Context, ev: QueryEvent) -> AnswerEvent:
        """Answer a sub-question using the appropriate tool."""
        print(f"Sub-question is {ev.question}")
        
        # Get the tools from the context
        tools = await ctx.get("tools")
        llm = await ctx.get("llm")
        
        # Use ReActAgent to handle the tools
        agent = ReActAgent.from_tools(tools, llm=llm, verbose=True)
        response = agent.chat(ev.question)
        
        return AnswerEvent(question=ev.question, answer=str(response))

    @step(pass_context=True)
    async def combine_answers(self, ctx: Context, ev: AnswerEvent) -> StopEvent | None:
        """Combine answers to sub-questions to form the final answer."""
        # Get the answers and sub-questions from the context
        answers = await ctx.get("answers") or []
        sub_questions = await ctx.get("sub_questions") or []
        
        # Add the new answer to the list
        answers.append({"question": ev.question, "answer": ev.answer})
        
        # Update the answers in the context
        await ctx.set("answers", answers)
        
        # If we have answers for all sub-questions, generate the final answer
        if len(answers) >= len(sub_questions):
            # Get the original query and LLM from the context
            original_query = await ctx.get("original_query")
            llm = await ctx.get("llm")
            
            # Format the answers for the prompt
            answers_text = "\n\n".join(
                [f"Question: {a['question']}\nAnswer: {a['answer']}" for a in answers]
            )
            
            # Generate the final answer
            prompt = f"""
            You are a helpful assistant that answers user questions based on the answers to sub-questions.
            
            Original question: {original_query}
            
            Sub-questions and answers:
            {answers_text}
            """
            
            print(f"Final prompt is {prompt}")
            
            response = llm.complete(prompt)
            
            print("Final response is", response)
            
            return StopEvent(result=str(response))
        
        # If we don't have answers for all sub-questions yet, continue
        # Get the next unanswered sub-question
        answered_questions = [a["question"] for a in answers]
        for question in sub_questions:
            if question not in answered_questions:
                return QueryEvent(question=question)
        
        # If all questions are answered but we didn't return a StopEvent earlier,
        # something went wrong, so return a default response
        return StopEvent(result="Failed to generate a complete answer.")
    
query_engine_tools = []

query_engine_tools.append(interest_rate_rag_tool)
query_engine_tools.append(customer_details_tool)
query_engine_tools.append(pending_tx_details_tool)

engine = SubQuestionQueryEngine(timeout=300, verbose=True)

async def main():
    result = await engine.run(
        llm=llm,
        tools=query_engine_tools,
        query="List all the details of Bob Brown?")
    
    print(result)

if __name__ == "__main__":
    asyncio.run(main())