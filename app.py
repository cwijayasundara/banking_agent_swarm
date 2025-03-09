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
import pandas as pd
# import streamlit as st

today = datetime.now().strftime("%d/%m/%Y")

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

query_engine = get_query_engine()
customer_db_query_engine = create_banking_customer_db()

llm = Gemini(model="models/gemini-2.0-flash",api_key=GOOGLE_API_KEY)

#  tools
async def search_interest_rates(ctx: Context, question: str) -> str:
    """Ask a question to the bank account interest rate documents stored in the vector index."""
    print("search interest rates tool called")
    interest_rates = query_engine.query(question)
    current_state = await ctx.get("state")
    # Store the interest rates in the state
    current_state["interest_rates"] = str(interest_rates)
    await ctx.set("state", current_state)
    return f"Interest rates extracted for {question}: {interest_rates}"

async def search_customer_details(ctx: Context, question: str) -> str:
    """Ask a question to the bank customer database which contains customer and account information in a SQL database."""
    print("search customer details tool called")
    customer_details = customer_db_query_engine.query(question)
    current_state = await ctx.get("state")
    # Store the customer details in the state
    current_state["customer_details"] = str(customer_details)
    await ctx.set("state", current_state)
    return f"Customer details extracted for {question}: {customer_details}"

async def search_pending_tx_details_from_df(ctx: Context, question: str) -> str:
    """Get the total amount of pending transactions for a customer from a Pandas dataframe."""
    print("search pending tx details tool called")
    response = get_pending_tx_details(question)
    current_state = await ctx.get("state")
    # Store the pending transactions details in the state
    current_state["pending_tx_details"] = str(response)
    await ctx.set("state", current_state)
    return f"Pending transactions details extracted for {question}: {response}"

# TODO: Add the overall analysis tool
async def overall_analysis(ctx: Context) -> str:
    """Get the overall analysis of the bank account for a customer from a Pandas dataframe."""
    print("overall analysis tool called")
    current_state = await ctx.get("state")
    
    interest_rates = None
    customer_details = None
    pending_tx_details = None

    # extract the interest rates if available
    if current_state["interest_rates"] is not None:
        interest_rates = current_state["interest_rates"]
        print("interest rates : ", interest_rates)

    # extract the customer details if available
    if current_state["customer_details"] is not None:
        customer_details = current_state["customer_details"]
        print("customer details : ", customer_details)
        
    # extract the pending transactions details if available
    if current_state["pending_tx_details"] is not None:
        pending_tx_details = current_state["pending_tx_details"]
        print("pending transactions details : ", pending_tx_details)
    
    # calculate the overall analysis using a LLM call
    prompt = f"""Based on the following information, please provide a detailed analysis of the bank account:
    
    Interest rates: {interest_rates}
    Customer details: {customer_details}
    Pending transactions details: {pending_tx_details}
    
    If any of the above information is not available, please mention that in your analysis.
    """
    
    overall_analysis = llm.complete(prompt)

    print("overall analysis : ", overall_analysis)
    
    # Store the overall analysis in the state
    current_state["overall_analysis"] = str(overall_analysis)
    await ctx.set("state", current_state)
    
    return f"Overall analysis completed: {overall_analysis}"

#  agents
supervisor_agent = FunctionAgent(
    name="SupervisorAgent",
    description = """Provide an overall analysis of the financial situation of a banking customer based on the data from InterestRatesAgent,CustomerDetailsAgent and PendingTxAgent. """,
    system_prompt = """You are an expert banking analyst and supervisor. You collect details coming from various agents such as InterestRatesAgent,CustomerDetailsAgent and PendingTxAgent."
         Details of the agents that you have access to is as follows.
         - InterestRatesAgent : This is a RAG agent that can search the bank account interest rate documents stored in the vector index. If the users question is related to the interest rates of the bank account like savings account, ISA account, etc.
         - CustomerDetailsAgent : This is an agent that can search the bank customer database which contains customer information in a SQL database. If the users question is related to the customer details like balance, name, address, bank balance, etc.
         - PendingTxAgent : This is a Pandas agent that can search the bank account pending transactions stored in a Pandas dataframe. If the users question is related to the pending transactions of the bank account like pending transactions amount, pending transactions date, etc.
         You should select the right agent based on the users question and pass the question to the right agent.
         If the user query is a complex query that needs to be answered by multiple agents, you should pass the question to all the agents and get the answers from all the agents.
         """,
    llm=llm,
    tools=[overall_analysis],
    can_handoff_to=["InterestRatesAgent", "CustomerDetailsAgent", "PendingTxAgent"],
)

interest_rates_agent = FunctionAgent(
    name="InterestRatesAgent",
    description="RAG agent that can search the bank account interest rate documents stored in the vector index.",
    system_prompt=(
        """You are the RAG Agent that can search the bank account interest rate documents stored in the vector index. Do not answer any other questions except for the ones related to the interest rates of the bank account."""
    ),
    llm=llm,
    tools=[search_interest_rates],
    can_handoff_to=["SupervisorAgent"],
)

customer_details_agent = FunctionAgent(
    name="CustomerDetailsAgent",
    description="Agent that can search the bank customer database which contains customer information in a SQL database.",
    system_prompt="You are the CustomerDetailsAgent that can search the bank customer database which contains customer information in a SQL database. Do not answer any other questions except for the ones related to the customer details of the bank account.",
    llm=llm,
    tools=[search_customer_details],
    can_handoff_to=["SupervisorAgent"],
)

pending_tx_agent = FunctionAgent(
    name="PendingTxAgent",
    description="Pandas agent that can search the bank account pending transactions stored in a Pandas dataframe.",
    system_prompt="You have access to a Pandas dataframe that contains details of pending transactions for a customer. Do not answer any other questions except for the ones related to the pending transactions of the bank account.",
    llm=llm,
    tools=[search_pending_tx_details_from_df],
    can_handoff_to=["SupervisorAgent"],
)

agent_workflow = AgentWorkflow(
    agents=[supervisor_agent, interest_rates_agent, customer_details_agent, pending_tx_agent],
    root_agent=supervisor_agent.name,
    initial_state={
        "interest_rates": {},
        "customer_details": {},
        "pending_tx_details": {},
        "overall_analysis": {},
    },
)

ctx = Context(agent_workflow)

async def main():
    question_1 = "Whats the Cash ISA Saver's annual interest rate for an account opened after 18/02/25? Todays date is " + today
    question_2 = "List all the details of Bob Brown?"
    question_3 = "What is the total amount of pending transactions for Bob Brown and round off to 2 decimal places?"

    response_1 = await agent_workflow.run(user_msg=question_1, ctx=ctx)
    response_2 = await agent_workflow.run(user_msg=question_2, ctx=ctx)
    response_3 = await agent_workflow.run(user_msg=question_3, ctx=ctx)
    
    print(response_1)
    print(response_2)
    print(response_3)

if __name__ == "__main__":
    asyncio.run(main())

# async def get_answer(query: str):
#     response = await agent_workflow.run(user_msg=query, ctx=ctx)
#     return response

# st.title("Personal Banking Assistant")

# with st.sidebar:
#     st.image("image/image.jpg", width=800)

# query = st.text_input("How can I help you today?")

# st.write("sample questions:")
# st.write("What is the total amount of pending transactions for customer id C001 and round off to 2 decimal places?")
# st.write("Whats the Cash ISA Saver's annual interest rate for an account opened after 18/02/25? Todays date is " + today)
# st.write("List all the details of Bob Brown?")

# if st.button("Chat"):
#     if query:
#         with st.spinner("Thinking..."):
#             # Use the helper function to run the async function
#             response = asyncio.run(get_answer(query))
#             st.write(response)
#     else:
#         st.warning("Please enter a query first.")
