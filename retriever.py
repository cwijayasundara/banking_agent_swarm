from dotenv import load_dotenv
import os
from llama_index.llms.gemini import Gemini
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.core import (
    StorageContext,
    load_index_from_storage,
)
from pathlib import Path
from llama_index.core.settings import Settings

load_dotenv()

persist_dir = "./vector_index"

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

llm = Gemini(model="models/gemini-2.0-flash-001",api_key=GOOGLE_API_KEY)

embed_model = GeminiEmbedding(model_name="models/text-embedding-004", api_key=GOOGLE_API_KEY)

Settings.llm = llm
Settings.embed_model = embed_model

def get_query_engine():
    """Initialize and return the query engine"""
    if not Path(persist_dir).exists():
        raise ValueError("No index found. Please ingest documents first.")
        
    storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
    index = load_index_from_storage(storage_context)
    return index.as_query_engine(similarity_top_k=5)

def search_documents(query: str) -> str:
    """Search through the bank account interest rate documents for relevant information."""
    query_engine = get_query_engine()
    response = query_engine.query(query)
    return str(response)

# query = "Whats the Cash ISA Saver's annual interest rate for an account opened after 18/02/25?"
# response = search_documents(query)
# print(response)