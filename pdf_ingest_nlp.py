# do standard NLP to extract data from full text and write into a new supabase table

#do only eb-1 cases

#part 2 of this will be the search py file that uses tantivy to search the data based on user query


from datetime import date as dt, datetime
from openai import OpenAI
from pydantic import BaseModel, Field, field_validator
import argparse
import fitz 
import instructor 
import nltk
import spacy
from typing import List
import lancedb
from lancedb.pydantic import LanceModel, vector
import pyarrow as pa
import json
import pandas as pd
from dotenv import load_dotenv
import os
import supabase
import uuid
import glob

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
GPT_MODEL = os.getenv("GPT_MODEL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

supabase = supabase.create_client(SUPABASE_URL, SUPABASE_KEY)


# Initialize the instructor client
instructor_client = instructor.from_openai(OpenAI())

#openai client
openai_client = OpenAI()

# Load the Spacy model
nlp = spacy.load("en_core_web_sm")

# Function to extract text from a PDF
def extract_text(pdf_path: str) -> str:
    """
    Extracts text from the PDF and returns it as a single string.
    """
    doc = fitz.open(pdf_path)
    full_text = [page.get_text("text") for page in doc]
    doc.close()
    return "\n".join(full_text)

def process_text(text: str) -> str:
    """
    remove punctuation
    remove whitespace, stop words
    convert to lowercase
    tokenizzation??
    normalization: include stemming and lemmatization
    handling contractions
    part of speech tagging
    Named entity recognition
    """


