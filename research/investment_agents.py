from dotenv import load_dotenv
import os
from financetoolkit import Toolkit
import pandas as pd
from llama_index.llms.gemini import Gemini
from llama_index.core.workflow import Context

load_dotenv()

FINANCIAL_MODELING_PREP_API_KEY = os.getenv('FINANCIAL_MODELING_PREP_API_KEY')
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')

llm = Gemini(model="models/gemini-2.0-flash",api_key=GOOGLE_API_KEY)

load_dotenv()

async def get_fundamentals(ctx: Context, ticker: str) ->pd.DataFrame(): # type: ignore
  """ Get the different fundamental ratios for a given ticker. """
  companies = Toolkit(
      [ticker], api_key=FINANCIAL_MODELING_PREP_API_KEY, start_date="2022-01-01"
  )
  ratios = companies.ratios.collect_all_ratios()

  print("ratios", ratios.loc['Return on Assets'])

  current_state = await ctx.get("state")

  if current_state["ticker"] == "":
    current_state["ticker"] = ticker

  if current_state["ratios"].empty:
    current_state["ratios"] = ratios

  await ctx.set("state", current_state)

  return f"Ratios extracted for {ticker}."


async def get_profitability_ratios(ctx: Context):
  """Get profitability ratios for a given ticker: ROA, ROE, Net Profit Margin and Gross Margin, and comments on these ratios given a set of threshold values."""

  current_state = await ctx.get("state")

  ratios = current_state['ratios']
  ticker = current_state['ticker']

  ROA = ratios.loc['Return on Assets']
  ROE = ratios.loc['Return on Equity']
  net_profit_margin = ratios.loc['Net Profit Margin']
  gross_margin = ratios.loc['Gross Margin']

  print("## Profitability Ratios (Assessing Earnings & Efficiency)")
  print("Return on Assets (ROA):", ROA.index[-1], ROA.iloc[-1])
  print("Return on Equity (ROE):", ROE.index[-1], ROE.iloc[-1])
  print("Net Profit Margin:", net_profit_margin.index[-1], net_profit_margin.iloc[-1])
  print("Gross Margin:", gross_margin.index[-1], gross_margin.iloc[-1])
  
  roa_values = [ROA.index[-1], ROA.iloc[-1]]
  roe_values = [ROE.index[-1], ROE.iloc[-1]]
  net_profit_margin_values = [net_profit_margin.index[-1], net_profit_margin.iloc[-1]]
  gross_margin_values = [gross_margin.index[-1], gross_margin.iloc[-1]]

  dico_ratios_profitability = {
      "Return on Assets (ROA)": roa_values,
      "Return on Equity (ROE)": roe_values,
      "Net Profit Margin": net_profit_margin_values,
      "Gross Margin": gross_margin_values
  }

  #Need to add conditions whether ratios, roa, roe, net profit margin, gross margin are empty or not
  if current_state["profitability_ratios"] == {}:
    current_state["profitability_ratios"] = dico_ratios_profitability

  print("current_state['profitability_ratios'] ==> after modifying state ==>", current_state["profitability_ratios"])

  thresholds_to_respect ="""
  ## Thresholds to respect for firm's financial health:
  ## Profitability Ratios (Assessing Earnings & Efficiency)
  |Ratio|	Healthy|	Moderate|	Weak|
  |-----|	-------|	--------|	----|
  |Return on Assets (ROA)	|> 5%|	2% - 5%	|< 2%|
  |Return on Equity (ROE)|	> 15%	|8% - 15%	|< 8%|
  |Net Profit Margin	|> 10%	|5% - 10%	|< 5%|
  |Gross Profit Margin	|> 40%	|20% - 40%	|< 20%|
  """

  prompt = f"""
  Analyze the financial health of the firm {ticker} based on its profitability ratios.

  ### Given:
  - **Profitability Ratios for {ticker}:**
    {dico_ratios_profitability}

  - **Thresholds for Financial Health Evaluation:**
    {thresholds_to_respect}

  ### Task:
  For each ratio, follow these steps:
  1️⃣ **Assign a score** from **1 to 10** (where **1 = very unhealthy**, **10 = very healthy**).
  2️⃣ **Provide a justification** explaining why the ratio received that score.
  3️⃣ **Give an overall insight** on the firm's financial health, summarizing strengths and weaknesses based on the individual ratio scores.

  Ensure the analysis is **detailed, data-driven, and easy to interpret**.
  """

  resp = llm.complete(prompt)
  current_state['threshold_profitability_comments'] = resp

  print('resp from LLLM', resp)

  await ctx.set("state", current_state)

  return "Profitability ratios extracted and Comments performed: " + str(resp)


async def get_liquidity_ratios(ctx: Context):
  """Get liquidity ratios for a given ticker: Current Ratio, Quick Ratio, Debt-to-Equity Ratio, Interest Coverage Ratio and comments on these ratios given a set of threshold values."""

  current_state = await ctx.get("state")
  ratios = current_state['ratios']
  ticker = current_state['ticker']

  current_ratio = ratios.loc['Current Ratio']
  quick_ratio = ratios.loc['Quick Ratio']
  debt_to_equity_ratio = ratios.loc['Debt-to-Equity Ratio']
  interest_coverage_ratio = ratios.loc['Interest Coverage Ratio']

  print("## Profitability Ratios (Assessing Earnings & Efficiency)")
  print("Current Ratio:", current_ratio.index[-1], current_ratio.iloc[-1])
  print("Quick Ratio:", quick_ratio.index[-1], quick_ratio.iloc[-1])
  print("Debt to Equity Ratio:", debt_to_equity_ratio.index[-1], debt_to_equity_ratio.iloc[-1])
  print("Interest Coverage Ratio:", interest_coverage_ratio.index[-1], interest_coverage_ratio.iloc[-1])
  current_ratio_values = [current_ratio.index[-1], current_ratio.iloc[-1]]
  quick_ratio_values = [quick_ratio.index[-1], quick_ratio.iloc[-1]]
  debt_to_equity_ratio_values = [debt_to_equity_ratio.index[-1], debt_to_equity_ratio.iloc[-1]]
  interest_coverage_ratio_values = [interest_coverage_ratio.index[-1], interest_coverage_ratio.iloc[-1]]

  dico_ratios_liquidity = {
      "Current Ratio": current_ratio_values,
      "Quick Ratio": quick_ratio_values,
      "Debt to Equity Ratio": debt_to_equity_ratio_values,
      "Interest Coverage Ratio": interest_coverage_ratio_values
  }

  #Need to add conditions whether ratios, roa, roe, net profit margin, gross margin are empty or not
  if current_state["liquidity_ratios"] == {}:
    current_state["liquidity_ratios"] = dico_ratios_liquidity

  print("current_state['liquidity_ratios'] ==> after modifying state ==>", current_state["liquidity_ratios"])

  thresholds_to_respect ="""
  ## Thresholds to respect for firm's financial health:
  ## Liquidity & Solvency Ratios (Assessing Financial Stability)
  | Ratio	| Healthy Range	 | Warning Zone	 | Risky/Dangerous  |
  | ----	| ------------	 | -----------	 | ---------------  |
  | Current Ratio	 |  1.5 - 3.0	 |  < 1.0	 | > 3.0 (excess cash)  |
  | Quick Ratio	 | > 1.0	 | < 1.0 | 	- |
  | Debt-to-Equity (D/E) | 0.3 - 1.5	 | > 2.0	 | < 0.3 (under-leveraged) |
  | Interest Coverage  | > 3.0	| 1.5 - 3.0	 | < 1.5 (high risk) |
  """

  prompt = f"""
  Analyze the financial health of the firm {ticker} based on its profitability ratios.

  ### Given:
  - **Profitability Ratios for {ticker}:**
    {dico_ratios_liquidity}

  - **Thresholds for Financial Health Evaluation:**
    {thresholds_to_respect}

  ### Task:
  For each ratio, follow these steps:
  1️⃣ **Assign a score** from **1 to 10** (where **1 = very unhealthy**, **10 = very healthy**).
  2️⃣ **Provide a justification** explaining why the ratio received that score.
  3️⃣ **Give an overall insight** on the firm's financial health, summarizing strengths and weaknesses based on the individual ratio scores.

  Ensure the analysis is **detailed, data-driven, and easy to interpret**.
  """

  resp = llm.complete(prompt)
  current_state['threshold_liquidity_comments'] = resp

  print('resp from LLLM', resp)

  await ctx.set("state", current_state)

  return "Liquidity ratios extracted and Comments performed: " + str(resp)


async def get_overall_comments(ctx: Context, overall_comments):
  """Get comments on diffrent type of ratios. The overall_comments are given by the SupervisedAgent based on threshold_profitability_comments and threshold_liquidity_comments."""

  current_state = await ctx.get("state")
  profitability_comments = current_state['threshold_profitability_comments']
  liquidity_comments = current_state['threshold_liquidity_comments']

  if profitability_comments is None:
    return "No profitability comments found."

  if liquidity_comments is None:
    return "No liquidity comments found."

  if ~ profitability_comments is None and ~ liquidity_comments is None:
    current_state['overall_comments'] = overall_comments

  return "Overall comments done."

# agents
from llama_index.core.agent.workflow import FunctionAgent, ReActAgent

fundamental_agent = FunctionAgent(
    name="FundamentalAgent",
    description="Get various fundamental ratios for a given ticker.",
    system_prompt=(
        "You are the fundament analyst that can extract different fundamental ratios for a given ticker. "
        "Once you have extracted the fundamental financial ratios, you should hand off control to the ProfitabilityAgent to extract the profitability ratios."
        "the ResearchAgent that can search the web for information on a given topic and record notes on the topic. "
    ),
    llm=llm,
    tools=[get_fundamentals],
    can_handoff_to=["ProfitabilityAgent"],
)

profitability_agent = FunctionAgent(
    name="ProfitabilityAgent",
    description="Collect profitability ratios for a given ticker: ROA, ROE, Net Profit Margin and Gross Margin and Comment on the results given a set of threshold values.",
    system_prompt=(
        """You are the ProfitabilityAgent that can collect profitability ratios (profitability_ratios) on a given ticker.
        You collect these ratios from the FundamentalAgent.
        Once these ratios are collected in profitability_ratios, you should comment on these ratios based on the thresholds values provided in get_profitability_ratios.
        These comments must be included in threshold_profitability_comments. At the end provide ONLY these comments included in threshold_profitability_comments. DO NOT ADD anything else.
        Once the comments are done, you should hand off control to the LiquidityAgent.
        """
    ),
    llm=llm,
    tools=[get_profitability_ratios],
    can_handoff_to=["LiquidityAgent"],
)

liquidity_agent = FunctionAgent(
    name="LiquidityAgent",
    description="Collect liquidity ratios for a given ticker: Current Ratio, Quick Ratio, Debt-to-Equity Ratio, Interest Coverage Ratio and comments on these ratios given a set of threshold values.",
    system_prompt=(
        """You are the LiquidityAgent that can collect liquidity ratios (liquidity_ratios) on a given ticker.
        You collect these ratios from the FundamentalAgent.
        Once these ratios are collected in liquidity_ratios, you should comment on these ratios based on the thresholds values provided in get_liquidity_ratios.
        These comments must be included in threshold_liquidity_comments. At the end provide ONLY these comments included in threshold_profitability_comments. DO NOT ADD anything else.
        Once the comments are done, you should hand off control to the SupervisorAgent.
        """
    ),
    llm=llm,
    tools=[get_liquidity_ratios],
    can_handoff_to=["SupervisorAgent"],
)

supervisor_agent = FunctionAgent(
    name="SupervisorAgent",
    description="Provide an overall comment based on the comments coming from the ProfitabilityAgent and LiquidityAgent.",
    system_prompt=(
        "You are an fundament analyst expert and supervisor. You collect comments coming from various agent such as ProfitabilityAgent and LiquidityAgent. "
        "Based on the results in the comments coming from the ProfitabilityAgent and LiquidityAgent, provide an overall comment on the health of the firm."
        "Justify your comment with details and data. "
    ),
    llm=llm,
    tools=[get_overall_comments],
    can_handoff_to=["FundamentalAgent"],
)

# workflow
from llama_index.core.agent.workflow import AgentWorkflow

agent_workflow = AgentWorkflow(
    agents=[fundamental_agent, profitability_agent,liquidity_agent, supervisor_agent],
    root_agent=fundamental_agent.name,
    initial_state={
        "ratios": pd.DataFrame(),
        "profitability_ratios": {},
        "liquidity_ratios": {},
        "threshold_profitability_comments": None,
        "threshold_liquidity_comments": None,
        "overall_comments": None,
        "ticker": "",
    },

)

from llama_index.core.agent.workflow import (
    AgentInput,
    AgentOutput,
    ToolCall,
    ToolCallResult,
    AgentStream,
)

# Main async function to run everything
async def main():
    # Create the workflow handler inside the async function
    handler = agent_workflow.run(
        user_msg=(
            "Provide the fundamental analysis of Nvidia and comments on the financial health of the company."
        )
    )

    current_agent = None
    current_tool_calls = ""

    # Process the events
    async for event in handler.stream_events():
        if (
            hasattr(event, "current_agent_name")
            and event.current_agent_name != current_agent
        ):
            current_agent = event.current_agent_name
            print(f"\n{'='*50}")
            print(f"🤖 Agent: {current_agent}")
            print(f"{'='*50}\n")

        elif isinstance(event, AgentOutput):
            if event.response.content:
                print("📤 Output:", event.response.content)
            if event.tool_calls:
                print(
                    "🛠️  Planning to use tools:",
                    [call.tool_name for call in event.tool_calls],
                )
        elif isinstance(event, ToolCallResult):
            print(f"🔧 Tool Result ({event.tool_name}):")
            print(f"  Arguments: {event.tool_kwargs}")
            print(f"  Output: {event.tool_output}")
        elif isinstance(event, ToolCall):
            print(f"🔨 Calling Tool: {event.tool_name}")
            print(f"  With arguments: {event.tool_kwargs}")

# Run the main function
if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
