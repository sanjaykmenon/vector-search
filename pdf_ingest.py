from datetime import date as dt
from openai import OpenAI
from pydantic import BaseModel, Field, field_validator
import argparse
import fitz 
import instructor 
import nltk
import spacy
from typing import List

# Initialize the instructor client
client = instructor.from_openai(OpenAI())

# Function to extract text from a PDF
def extract_text(pdf_path: str) -> str:
    """
    Extracts text from the PDF and returns it as a single string.
    """
    doc = fitz.open(pdf_path)
    full_text = [page.get_text("text") for page in doc]
    doc.close()
    return "\n".join(full_text)

# Pydantic model for the initial summary
class InitialSummary(BaseModel):
    """
    This is an initial summary which should be long (4-5 sentences, ~80 words)
    yet highly non-specific, containing little information beyond the entities marked as missing.
    Use overly verbose languages and fillers (Eg. This article discusses) to reach ~80 words.
    """
    summary: str = Field(
        ...,
        description="This is a summary of the article provided which is overly verbose and uses fillers. It should be roughly 80 words in length",
    )
    
class RewrittenSummary(BaseModel):
    """

    This is a new, denser summary of identical length which covers every entity
    and detail from the previous summary plus the Missing Entities.

    Guidelines
    - Make every word count : Rewrite the previous summary to improve flow and make space for additional entities
    - Never drop entities from the previous summary. If space cannot be made, add fewer new entities.
    - The new summary should be highly dense and concise yet self-contained, eg., easily understood without the Article.
    - Make space with fusion, compression, and removal of uninformative phrases like "the article discusses"
    - Missing entities can appear anywhere in the new summary

    An Entity is a real-world object that's assigned a name - for example, a person, country a product or a book title.
    """
    summary: str = Field(
        ...,
        description="This is a new, denser summary of identical length which covers every entity and detail from the previous summary plus the Missing Entities. It should have the same length ( ~ 80 words ) as the previous summary and should be easily understood without the Article",
    )
    absent: List[str] = Field(
        ...,
        default_factory=list,
        description="this is a list of Entities found absent from the new summary that were present in the previous summary",
    )
    missing: List[str] = Field(
        default_factory=list,
        description="This is a list of 1-3 informative Entities from the Article that are missing from the new summary which should be included in the next generated summary.",
    )

    @field_validator("summary")
    def min_length(cls, v: str) -> str:
        """
        Validates that the summary is at least 60 words long.
        """
        tokens = nltk.word_tokenize(v)
        num_tokens = len(tokens)
        if num_tokens < 60:
            raise ValueError("The current summary is too short. Please make sure that you generate a new summary that is around 80 words long.")
        return v

    @field_validator("summary")
    def min_entity_density(cls, v: str) -> str:
        """
        Validates that the summary has a sufficient entity density.
        """
        tokens = nltk.word_tokenize(v)
        num_tokens = len(tokens)
        doc = nlp(v)
        num_entities = len(doc.ents)
        density = num_entities / num_tokens
        if density < 0.08:
            raise ValueError(f"The summary of {v} has too few entities. Please regenerate a new summary with more new entities added to it. Remember that new entities can be added at any point of the summary.")
        return v

# Load the Spacy model
nlp = spacy.load("en_core_web_sm")

def summarize_article(article: str, summary_steps: int = 3):
    summary_chain = []

    #generating initial summary
    summary: InitialSummary = client.chat.completions.create(
        model="gpt-3.5-turbo-0125",
        response_model=InitialSummary,
        messages=[
            {
                "role": "system",
                "content": "Write a summary about the article that is long (4-5 sentences) yet highly non-specific. Use overly, verbose language and fillers(eg.,'this article discusses') to reach ~80 words"
            },
            {
                "role":"user", "content": f"Here is the Article: {article}"},
            {
                "role":"user",
                "content": "The generated summary should be about 80 words",
            },
        ],
        max_retries=2,
    )
    summary_chain.append(summary.summary)
    print(f"Initial Summary:\n{summary.summary}\n")

    prev_summary = None
    summary_chain.append(summary.summary)
    for i in range(summary_steps):
        missing_entity_message = (
            []
            if prev_summary is None
            else [
                {
                    "role": "user",
                    "content": f"Please include these Missing Entities: {','.join(prev_summary.missing)}",
                },
            ]
        )
        new_summary: RewrittenSummary = client.chat.completions.create( 
            model="gpt-3.5-turbo-0125",
            messages=[
                {
                    "role": "system",
                    "content": """
                You are going to generate an increasingly concise,entity-dense summary of the following article.

                Perform the following two tasks
                - Identify 1-3 informative entities from the following article which is missing from the previous summary
                - Write a new denser summary of identical length which covers every entity and detail from the previous summary plus the Missing Entities

                Guidelines
                - Make every word count: re-write the previous summary to improve flow and make space for additional entities
                - Make space with fusion, compression, and removal of uninformative phrases like "the article discusses".
                - The summaries should become highly dense and concise yet self-contained, e.g., easily understood without the Article.
                - Missing entities can appear anywhere in the new summary
                - Never drop entities from the previous summary. If space cannot be made, add fewer new entities.
                """,
                },
                {"role": "user", "content": f"Here is the Article: {article}"},
                {
                    "role": "user",
                    "content": f"Here is the previous summary: {summary_chain[-1]}",
                },
                *missing_entity_message,
            ],
            max_retries=3, 
            max_tokens=1000,
            response_model=RewrittenSummary,
        )
        summary_chain.append(new_summary.summary)
        print(f"Summary Revision {i+1}:\n{new_summary.summary}\n") 
        prev_summary = new_summary
    return summary_chain

#TODO convert summary as structured output from LLM including following
# date, beneficiary details, petitioner details, and key points / reasons.


def main(pdf_path: str):
    text = extract_text(pdf_path)
    json_response = summarize_article(text)
    print(json_response)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract and process PDF text into structured JSON data.")
    parser.add_argument('pdf_path', type=str, help='Path to the PDF file')
    args = parser.parse_args()
    main(args.pdf_path)
