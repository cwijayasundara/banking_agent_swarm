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
from pending_tx_agent import get_pending_tx_details

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

def get_pending_tx_details_from_df(question: str) -> str:
    """Get the total amount of pending transactions for a customer from a Pandas dataframe."""
    response = get_pending_tx_details(question)
    return str(response)

#  agents

supervisor_agent = ReActAgent(
    name="SupervisorAgent",
    description="You are the supervisor agent that oversees the work of the interest rates agent and the customer details agent and decide which agent to hand off control to.",
    system_prompt=(
        "Based on users question and based on the agents that are available to you, you should plan how to utilise these agents to answer the users question."
        "You should have at least some notes on a topic before handing off control to the interest rates agent or the customer details agent."
        "If the answer to the users question needs to be derived from the interest rates agent and the customer details agent both, then you should hand off control to both agents, and then combine the results into a single answer."
    ),
    llm=llm_thinking,
    tools=[],
    can_handoff_to=["InterestRatesAgent", "CustomerDetailsAgent", "PendingTxAgent"],
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

pending_tx_agent = FunctionAgent(
    name="PendingTxAgent",
    description="Useful for getting details of pending transactions for a customer from a Pandas dataframe.",
    system_prompt="You have access to a Pandas dataframe that contains details of pending transactions for a customer."
    "You can use this agent to get details of pending transactions for a customer.",
    llm=llm,
    tools=[get_pending_tx_details_from_df],
)

agent_workflow = AgentWorkflow(
    agents=[supervisor_agent, interest_rates_agent, customer_details_agent, pending_tx_agent],
    root_agent=supervisor_agent.name,
    initial_state={
        
    },
)

ctx = Context(agent_workflow)

async def main():
    question_1 = "Whats the Cash ISA Saver's annual interest rate for an account opened after 18/02/25? Todays date is " + today
    question_2 = "What is the customer name for customer id C001?"
    question_3 = "What is the total amount of pending transactions for customer id C001 and round off to 2 decimal places?"
    response_1 = await agent_workflow.run(user_msg=question_1, ctx=ctx)
    response_2 = await agent_workflow.run(user_msg=question_2, ctx=ctx)
    response_3 = await agent_workflow.run(user_msg=question_3, ctx=ctx)
    print(response_1)
    print(response_2)
    print(response_3)

if __name__ == "__main__":
    asyncio.run(main())
