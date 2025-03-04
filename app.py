import asyncio
from dotenv import load_dotenv
import os
from retriever import get_query_engine
from llama_index.core.agent.workflow import AgentWorkflow
from llama_index.llms.gemini import Gemini
from datetime import datetime
from llama_index.core.workflow import Context
from llama_index.core.agent.workflow import FunctionAgent, ReActAgent
from llama_index.core.agent.workflow import AgentWorkflow
from customer_db import create_banking_customer_db

today = datetime.now().strftime("%d/%m/%Y")

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

query_engine = get_query_engine()

customer_db_query_engine = create_banking_customer_db()

llm = Gemini(model="models/gemini-2.0-flash-001",api_key=GOOGLE_API_KEY)

llm_thinking = Gemini(model="models/gemini-2.0-flash-thinking-exp-01-21",api_key=GOOGLE_API_KEY, temperature=1.0)

#  tools
def search_interest_rates(question: str) -> str:
    """Ask a question to the bank account interest rate documents stored in the vector index."""
    response = query_engine.query(question)
    return str(response)

def search_customer_details(question: str) -> str:
    """Ask a question to the bank customer database which contains customer information in a SQL database."""
    response = customer_db_query_engine.query(question)
    return str(response)

#  agents

supervisor_agent = ReActAgent(
    name="SupervisorAgent",
    description="You are the supervisor agent that can oversee the work of the interest rates agent and the customer details agent and decide which agent to hand off control to.",
    system_prompt=(
        "You are the SupervisorAgent that can oversee the work of the interest rates agent and the customer details agent. You can hand off control to the interest rates agent or the customer details agent when you have enough information to answer the user's question."
    ),
    llm=llm_thinking,
    tools=[],
    can_handoff_to=["InterestRatesAgent", "CustomerDetailsAgent"],
)

interest_rates_agent = FunctionAgent(
    name="InterestRatesAgent",
    description="Useful for searching the bank account interest rate documents stored in the vector index.",
    system_prompt=(
        "You are the RagAgent that can search the bank account interest rate documents stored in the vector index. "
    ),
    llm=llm,
    tools=[search_interest_rates],
)

customer_details_agent = FunctionAgent(
    name="CustomerDetailsAgent",
    description="Useful for searching the bank customer database which contains customer information in a SQL database.",
    system_prompt="You are the CustomerDetailsAgent that can search the bank customer database which contains customer information in a SQL database.",
    llm=llm,
    tools=[search_customer_details],
)

agent_workflow = AgentWorkflow(
    agents=[supervisor_agent, interest_rates_agent, customer_details_agent],
    root_agent=supervisor_agent.name,
    initial_state={
        
    },
)

ctx = Context(agent_workflow)

async def main():
    question_1 = "Whats the Cash ISA Saver's annual interest rate for an account opened after 18/02/25? Todays date is " + today
    question_2 = "What is the customer name for customer id C001?"
    response_1 = await agent_workflow.run(user_msg=question_1, ctx=ctx)
    response_2 = await agent_workflow.run(user_msg=question_2, ctx=ctx)
    print(response_1)
    print(response_2)

if __name__ == "__main__":
    asyncio.run(main())
