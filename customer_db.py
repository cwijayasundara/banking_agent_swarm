from llama_index.core import SQLDatabase, Settings
from llama_index.llms.gemini import Gemini
from sqlalchemy import (
    create_engine,
    MetaData,
    Table,
    Column,
    String,
    Integer,
)
from llama_index.core.query_engine import NLSQLTableQueryEngine
from sqlalchemy import insert
from dotenv import load_dotenv
import os

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

def create_banking_customer_db():
    """
    Creates and populates a banking customer database, returns the query engine
    """
    Settings.llm = Gemini(model="models/gemini-2.0-pro-exp-02-05",api_key=GOOGLE_API_KEY)
    engine = create_engine("sqlite:///:memory:", future=True)
    metadata_obj = MetaData()

    customer_table = Table(
        "customer",
        metadata_obj,
        Column("customer_id", String(16), primary_key=True),
        Column("customer_name", String(50), nullable=False),
        Column("customer_address", String(50), nullable=False),
        Column("customer_phone", String(50)),
        Column("customer_email", String(50)),
        Column("customer_dob", String(50)),
        Column("customer_gender", String(50)),
        Column("customer_nationality", String(50)),
        Column("customer_occupation", String(50)),
        Column("customer_income", String(50)),
        Column("account_balance", String(50)),
        
    )

    metadata_obj.create_all(engine)

    # Insert data into the customer table. Customers are from the UK

    rows = [
        {"customer_id": "C001", "customer_name": "John Smith", "customer_address": "123 Oak St, London, UK", "customer_phone": "123-456-7890", "customer_email": "john.smith@example.com", "customer_dob": "1980-01-15", "customer_gender": "Male", "customer_nationality": "UK", "customer_occupation": "Engineer", "customer_income": "£60000", "account_balance": "£15000"},
        {"customer_id": "C003", "customer_name": "Alice Johnson", "customer_address": "789 Pine St, Anytown, UK", "customer_phone": "123-456-7892", "customer_email": "alice.johnson@example.com", "customer_dob": "1992-03-20", "customer_gender": "Female", "customer_nationality": "UK", "customer_occupation": "Teacher", "customer_income": "£40000", "account_balance": "£8000"},
        {"customer_id": "C004", "customer_name": "Bob Brown", "customer_address": "123 Main St, Anytown, UK", "customer_phone": "123-456-7893", "customer_email": "bob.brown@example.com", "customer_dob": "1985-05-15", "customer_gender": "Male", "customer_nationality": "UK", "customer_occupation": "Doctor", "customer_income": "£70000", "account_balance": "£25000"},
        {"customer_id": "C005", "customer_name": "Charlie Davis", "customer_address": "456 Oak Ave, Anycity, UK", "customer_phone": "123-456-7894", "customer_email": "charlie.davis@example.com", "customer_dob": "1990-01-01", "customer_gender": "Male", "customer_nationality": "UK", "customer_occupation": "Software Engineer", "customer_income": "£50000", "account_balance": "£12000"},
        {"customer_id": "C006", "customer_name": "Tom Johns", "customer_address": "789 Pine St, Anytown, UK", "customer_phone": "123-456-7895", "customer_email": "tom.johns@example.com", "customer_dob": "1992-03-20", "customer_gender": "Male", "customer_nationality": "UK", "customer_occupation": "Teacher", "customer_income": "£40000", "account_balance": "£7500"},
        {"customer_id": "C007", "customer_name": "Jane Fonda", "customer_address": "123 Main St, Anytown, UK", "customer_phone": "123-456-7896", "customer_email": "jane.fonda@example.com", "customer_dob": "1985-05-15", "customer_gender": "Male", "customer_nationality": "UK", "customer_occupation": "Doctor", "customer_income": "£70000", "account_balance": "£30000"},
    ]

    for row in rows:
        stmt = insert(customer_table).values(**row)
        with engine.begin() as connection:
            connection.execute(stmt)

    sql_database = SQLDatabase(engine, include_tables=["customer"])
    
    return NLSQLTableQueryEngine(
        sql_database=sql_database,
        tables=["customer"]
    )

# Example usage:
# if __name__ == "__main__":
#     query_engine = create_banking_customer_db()
#     # Use a clear natural language query format
#     query = "List all the details of Bob Brown?"
#     response = query_engine.query(query)
#     print(response)