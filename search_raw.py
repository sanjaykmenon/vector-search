#write a python program to connect to a supabase table and run a select quer
import os
import sys
import json
import requests
import psycopg2
from psycopg2 import sql
import instructor
import supabase
import openai
from openai import OpenAI
from pydantic import BaseModel, Field, field_validator
from dotenv import load_dotenv
import openai

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
GPT_MODEL = os.getenv("GPT_MODEL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")

supabase = supabase.create_client(SUPABASE_URL, SUPABASE_KEY)


# Initialize the instructor client
instructor_client = instructor.from_openai(OpenAI())

#openai client
openai_client = OpenAI()


# Define your PostgreSQL database connection parameters
db_params = {
    'dbname': DB_NAME,
    'user': DB_USER,
    'password': DB_PASSWORD,
    'host': DB_HOST,
    'port': DB_PORT
}

# Define the query embedding, match threshold, and match count

match_threshold = 1.0
match_count = 5


query = sql.SQL("""
    SELECT *
    FROM match_documents(
        %s::vector(1536),  -- pass the query embedding
        %s,  -- match threshold
        %s   -- match count
    )
""")

def generate_openai_embedding(text: str) -> list:
    try:
        response = openai_client.embeddings.create(
            input=text,
            model="text-embedding-3-small"
        )
        return response.data[0].embedding
    
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return []

def search_database(query_embedding: list, match_threshold: float, match_count: int) -> list:
    
    try:
        with psycopg2.connect(**db_params) as conn:
            with conn.cursor() as cur:
                cur.execute(query, (query_embedding, match_threshold, match_count))
                results = cur.fetchall()
        return results
    
    except Exception as e:
        print(f"Database search error: {e}")
        return []
    
# the idea here is to have an assistant that provides an analysis as an immigration 
# expert with the context provided.

def get_llm_response(context: str, user_query: str) -> str:
    #prompt = f"Based on the following context, as an expert immigration attorney, answer the question. You may have some context that is not relevant to the question, ignore those, and only use what is necessary:\n\n{context}\n {user_query}?"
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o-2024-05-13",
            messages=[
        {
            "role": "system",
            "content": "You are an expert immigration attorney and an expert in immigration e-discovery. Answer the following question based on the provided context. Ignore any irrelevant information and only use what is necessary to provide a comprehensive response. Provide details on reasons for meeting / not meeting any criteria, analysis and don't make them verbose.Explain each case on their individual merits"
        },
        {
            "role": "user",
            "content": f"{context}\n\n{user_query}?"
        }
    ],
            max_tokens=4000
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error generating LLM response: {e}")
        return ""

def main():

    if len(sys.argv) < 2:
        print("Please provide a user query as a command line argument.")
        return

    user_query = sys.argv[1]

    query_embedding = generate_openai_embedding(user_query)

    if not query_embedding:
        print("Failed to generate embedding for the user query.")
        return
    else:
        print("Embedding generated successfully.")
        #print(query_embedding)
    
        match_threshold = 1  # Example threshold
        match_count = 5  # Example count

        results = search_database(query_embedding, match_threshold, match_count)
        if not results:
            print("No results found in the database.")
            return
        else:
            print("Results found in the database.")
            answer = get_llm_response(results, user_query)
            print(answer)


if __name__ == "__main__":
    main()


# TODO: Implement continuous questions chat for follow-ups.
# TODO: original rag context should provide uid or metadata for further querying  / follow-up questions.
# TODO: Provide full text / unique ID of a specific case in the original context for use by the LLM to retrieve the whole text for further analysis.
# TODO: Allow getting user input of a specific case in the first response and obtaining more details on that programmatically.
# TODO: Implement functionality for EB-2 NIW and EB-1 categories.
# TODO: Populate the database with 100 records of each case type for an MVP.
# TODO: Create an API using FastAPI for the entire application.
# TODO: Use Docker to containerize the application.
# TODO: Create a basic front end (details to be hashed out).
# TODO: For the frontend MVP, use Streamlit or something easy and simple.
# TODO: Get input from Lorenz on the first-gen app.
