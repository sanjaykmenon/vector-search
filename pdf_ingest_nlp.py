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

# Function to extract text from a PDF
def extract_text(pdf_path: str) -> str:
    """
    Extracts text from the PDF and returns it as a single string.
    """
    doc = fitz.open(pdf_path)
    full_text = [page.get_text("text") for page in doc]
    doc.close()
    return "\n".join(full_text)

#modify class Reason for all individual criteria and include CFR code and evidence
#add specific attribute for each criteria
#add class that extracts job title. 
# evals/ validation??

class Reason(BaseModel):
    reason: str = Field(..., description="The individual reason for approval or denial")
    cfr_code: str = Field(..., description="The respective CFR code")
    evidence: str = Field(..., description="The evidence provided for the reason")

class DocumentInfo(BaseModel):
    title: str = Field(..., description="The title of the document with specific details")
    beneficiary_details: List[str] = Field(..., description="extract details of beneficiary such as where they are from, what did they do and anything else that can be used to uniquely identify them")
    beneficiary_status: str = Field(..., description="extract details of the type of visa / status")
    key_reasons: List[Reason] = Field(..., description="extract the individual reasons for approval or denial, the respective CFR codes, and the evidence provided for each reason. ")
    summary: List[str] = Field(..., description="add details of entites, people, locations and any other specific detail")
    date_of_application: dt = Field(..., description="extract date of the document")
    summary_embedding: List[float] = Field(..., description="OpenAI embedding of the summary")
    footnotes: List[str] = Field(..., description="extract all footnotes mentioned in the text.")
    cfr_code: List[str] = Field(..., description="Extract all the sections of the U.S. Code and the Code of Federal Regulations mentioned in the text. ")

    summary: str = None
    full_text: str = None

    def set_full_text(self, full_text: str):
        self.full_text = full_text
    
    def set_summary(self, summary: str):
        self.summary = summary

    def set_summary_embedding(self, summary_embedding: List[float]):
        self.summary_embedding = summary_embedding

# Load the Spacy model
nlp = spacy.load("en_core_web_sm")