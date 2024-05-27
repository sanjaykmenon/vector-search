# test py

# write a function that takes a word and converts it to the text-embedding-small embeding using openai and print it

from openai import OpenAI
import os

openai_client = OpenAI()

def get_openai_key():
    return os.environ.get("OPENAI_API_KEY")

def generate_openai_embedding(text: str):
    response = openai_client.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )
    return response.data[0].embedding

def main():
    word = "opera singer"
    print(generate_openai_embedding(word))

if __name__ == "__main__":
    main()

# python3 test.py