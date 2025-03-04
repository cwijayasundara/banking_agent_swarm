import os
from dotenv import load_dotenv
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_parse import LlamaParse
from llama_index.core import VectorStoreIndex
from llama_index.core.schema import TextNode
from typing import List
import json

load_dotenv()

file_paths = ["docs/interest-rates_1.pdf", "docs/interest-rates_2.pdf", "docs/interest-rates_3.pdf"]

persist_dir = "./vector_index"

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

model_name = "models/text-embedding-004"

embed_model = GeminiEmbedding(model_name=model_name, api_key=GOOGLE_API_KEY)

system_prompt = """
You are given bank interest rates in text and tables format for a UK bank.
Make sure to parse out the text and tables correctly.
"""

parser = LlamaParse(
    result_type="markdown",
    use_vendor_multimodal_model=True,
    vendor_multimodal_model_name="gemini-2.0-flash-001",
    invalidate_cache=True,
    system_prompt=system_prompt,
)
    
def ingest_pdf(file_paths: List[str], persist_dir: str):
    """
    Ingest a PDF file and push it to the vector index
    """
    print("pushing the document to the vector index")

    documents = parser.load_data(file_paths)

    index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)
    index.storage_context.persist(persist_dir=persist_dir)
    return index

# Ingest the PDF files into the vector index
# index = ingest_pdf(file_paths, persist_dir)
# question = "Whats the Cash ISA Saver's annual interest rate for an account opened after 18/02/25?"
# response = index.as_query_engine().query(question)
# print(response)

