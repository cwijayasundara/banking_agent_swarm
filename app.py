import asyncio
from dotenv import load_dotenv
import os
from retriever import get_query_engine
from llama_index.core.agent.workflow import AgentWorkflow
from llama_index.core.settings import Settings
from llama_index.llms.gemini import Gemini
from datetime import datetime
from llama_index.core.workflow import Context
from llama_index.core.agent.workflow import FunctionAgent, ReActAgent
from llama_index.core.agent.workflow import AgentWorkflow

today = datetime.now().strftime("%d/%m/%Y")

load_dotenv()

query_engine = get_query_engine()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

llm = Gemini(model="models/gemini-2.0-flash-001",api_key=GOOGLE_API_KEY)

Settings.llm = llm

#  tools
def rag_tool(question: str) -> str:
    """Ask a question to the bank account interest rate documents stored in the vector index."""
    response = query_engine.query(question)
    return str(response)

#  agents
rag_agent = FunctionAgent(
    name="RagAgent",
    description="Useful for searching the bank account interest rate documents stored in the vector index.",
    system_prompt=(
        "You are the RagAgent that can search the bank account interest rate documents stored in the vector index. "
    ),
    llm=llm,
    tools=[rag_tool],
)


agent_workflow = AgentWorkflow(
    agents=[rag_agent],
    root_agent=rag_agent.name,
    initial_state={
        
    },
)

ctx = Context(agent_workflow)

async def main():
    question = "Whats the Cash ISA Saver's annual interest rate for an account opened after 18/02/25? Todays date is " + today
    response = await agent_workflow.run(user_msg=question, ctx=ctx)
    print(response)

if __name__ == "__main__":
    asyncio.run(main())
