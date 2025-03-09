from dotenv import load_dotenv
import os
import pandas as pd
from llama_index.experimental.query_engine import PandasQueryEngine
from llama_index.llms.gemini import Gemini

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

llm = Gemini(model="models/gemini-2.0-pro-exp-02-05",api_key=GOOGLE_API_KEY)

df = pd.read_csv("docs/pending_tx.csv")

def create_pending_tx_query_engine():
    query_engine = PandasQueryEngine(df=df, verbose=True)
    return query_engine

def get_pending_tx_details(query: str) -> float:
    query_engine = create_pending_tx_query_engine()
    response = query_engine.query(query)
    return response

# query = "What is the total amount of pending transactions for customer id C001 and round off to 2 decimal places?"
# print(get_pending_tx_details(query))