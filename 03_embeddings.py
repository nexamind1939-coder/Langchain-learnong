import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

# Load API key from .env file
load_dotenv(override=True)

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("OPENAI_API_KEY is missing from .env!")

# Create OpenAI embeddings
openai_embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=api_key
)

# Test embedding
result = openai_embeddings.embed_query(
    "LangChain helps build AI applications."
)

print("OpenAI embedding successful!")
print("Vector dimensions:", len(result))
text = "I love programming in Python."

vector = openai_embeddings.embed_query(text)

print("How many numbers in this vector?", len(vector))
print("First 5 numbers:", vector[:5])
text = "LangChain helps build AI applications."

vector = openai_embeddings.embed_query(text)

print("Embedding dimensions:", len(vector))
print("First 5 values:", vector[:5])

import numpy as np

def similarity(a, b):
    a, b = np.array(a), np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

v1 = openai_embeddings.embed_query("I like dogs")
v2 = openai_embeddings.embed_query("I love puppies")
v3 = openai_embeddings.embed_query("The car is fast")

print("dogs vs puppies (similar):", round(similarity(v1, v2), 2))
print("dogs vs car   (different):", round(similarity(v1, v3), 2))