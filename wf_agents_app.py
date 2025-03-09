from wf_agents import CustomerInvestmentAdvisorAgent, interest_rate_rag_tool, customer_details_tool, pending_tx_details_tool
import asyncio  

async def get_answer(query):
    agent = CustomerInvestmentAdvisorAgent(timeout=600, verbose=True)
    handler = agent.run(
        query=query,
        tools=[interest_rate_rag_tool, customer_details_tool, pending_tx_details_tool],
    )
    final_result = await handler
    print(final_result)


if __name__ == "__main__":
    asyncio.run(get_answer("Whats the Cash ISA Saver's annual interest rate for an account opened after 18/02/25?"))
