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
import re

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
    
def get_metadata_from_db(metadata_id: list):
    try:
        with psycopg2.connect(**db_params) as conn:
            with conn.cursor() as cur:
                query = "SELECT a.id, a.full_text FROM oa1_aao as a WHERE id = ANY(%s::uuid[])"
                cur.execute(query, (metadata_id,))
                metadata = cur.fetchall() #maybe one initially and all later?
        return metadata
    
    except Exception as e:
        print(f"Error retrieving full case details: {e}")
        return None

def extract_uuids_from_response(response: str) -> list:
    # Regular expression to match all UUIDs
    uuid_pattern = r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b"
    return re.findall(uuid_pattern, response)

    
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
            "content": "You are an expert immigration attorney and an expert in immigration e-discovery. Answer the following question based on the provided context. Ignore any irrelevant information and only use what is necessary to provide a comprehensive response. Provide details on reasons for meeting / not meeting any criteria, analysis and don't make them verbose.Explain each case on their individual merits and provide their individual uuid"
        },
        {
            "role": "user",
            "content": f"{context}\n\n{user_query}?"
        }
    ],
            max_tokens=4000
        )
        llm_response = response.choices[0].message.content.strip()
        return llm_response

    except Exception as e:
        print(f"Error working with OpenAI: {e}")
        return None
    
def follow_up_response(metadata_info: str) -> str:
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o-2024-05-13",
            messages=[
        {
            "role": "system",
            "content": "You are an expert immigration attorney. Reflect, think step by step and provide your expert analysis on these AAO decisions. Explain each case on their individual merits focusing on each criteri separately and its respective AAO decision along with your analysis and not in a combined manner."
        },
        {
                        "role": "user",
                        "content": f"{metadata_info}"
        }
    ],
                max_tokens=4000
            )
        follow_up_response = response.choices[0].message.content.strip()
        return follow_up_response
    
    except Exception as e:
        print(f"Error working with OpenAI follow up response: {e}")
        return None

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
    
    follow_up  = "Are you interested in more details on this case or do you have any other questions?"

    print(f"{answer}\n\n{follow_up}")

    user_input = input("Please enter your response: ")

    if user_input.lower() in ["yes", "y"]:
        # Extract all UUIDs from the LLM response
        metadata_ids = extract_uuids_from_response(answer)
        print(metadata_ids)
        metadata_results = []
        if metadata_ids:
            metadata = get_metadata_from_db(metadata_ids)
            #print(metadata)
            if metadata:
                for id, full_text in metadata:
                    metadata_results.append(f"Metadata for {id}: {full_text}")
            else:
                metadata_results.append("No additional information available for the given cases.")

        metadata_info = "\n\n".join(metadata_results)
        #print(metadata_info)
        follow_up_answer = follow_up_response(metadata_info)
        print(follow_up_answer)
    else:
        return "Thanks for not answering yes"
    

if __name__ == "__main__":
    main()


# TODO: original rag context should provide uid or metadata for further querying  / follow-up questions.
    # TODO: modify match_docs to return metadata for each document.
    # TODO: return data only useful for context for first llm response, but can be retrieved in same session later.
# TODO: Implement continuous questions chat for follow-ups.

# TODO: Provide full text / unique ID of a specific case in the original context for use by the LLM to retrieve the whole text for further analysis.
# TODO: Allow getting user input of a specific case in the first response and obtaining more details on that programmatically.
# TODO: Implement functionality for EB-2 NIW and EB-1 categories.
# TODO: Populate the database with 100 records of each case type for an MVP.
# TODO: Create an API using FastAPI for the entire application.
# TODO: Use Docker to containerize the application.
# TODO: Create a basic front end (details to be hashed out).
# TODO: For the frontend MVP, use Streamlit or something easy and simple.
# TODO: Get input from Lorenz on the first-gen app.
