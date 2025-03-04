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
    Settings.llm = Gemini(model="models/gemini-2.0-flash-001",api_key=GOOGLE_API_KEY)
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
        {"customer_id": "C001", "customer_name": "Alice Johnson", "customer_address": "789 Pine St, Anytown, UK", "customer_phone": "123-456-7892", "customer_email": "alice.johnson@example.com", "customer_dob": "1992-03-20", "customer_gender": "Female", "customer_nationality": "UK", "customer_occupation": "Teacher", "customer_income": "£40000", "account_balance": "£1000"},
        {"customer_id": "C002", "customer_name": "Bob Brown", "customer_address": "123 Main St, Anytown, UK", "customer_phone": "123-456-7893", "customer_email": "bob.brown@example.com", "customer_dob": "1985-05-15", "customer_gender": "Male", "customer_nationality": "UK", "customer_occupation": "Doctor", "customer_income": "£70000", "account_balance": "£2000"},
        {"customer_id": "C003", "customer_name": "Charlie Davis", "customer_address": "456 Oak Ave, Anycity, UK", "customer_phone": "123-456-7894", "customer_email": "charlie.davis@example.com", "customer_dob": "1990-01-01", "customer_gender": "Male", "customer_nationality": "UK", "customer_occupation": "Software Engineer", "customer_income": "£50000", "account_balance": "£3000"},
        {"customer_id": "C004", "customer_name": "Tom Johns", "customer_address": "789 Pine St, Anytown, UK", "customer_phone": "123-456-7895", "customer_email": "tom.johns@example.com", "customer_dob": "1992-03-20", "customer_gender": "Male", "customer_nationality": "UK", "customer_occupation": "Teacher", "customer_income": "£40000", "account_balance": "£4000"},
        {"customer_id": "C005", "customer_name": "Jane Fonda", "customer_address": "123 Main St, Anytown, UK", "customer_phone": "123-456-7896", "customer_email": "jane.fonda@example.com", "customer_dob": "1985-05-15", "customer_gender": "Male", "customer_nationality": "UK", "customer_occupation": "Doctor", "customer_income": "£70000", "account_balance": "£5000"},
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
#     # Use a more explicit query format
#     query = "Show me all information about the customer with customer_id equal to C003"
#     response = query_engine.query(query)
#     print(response)