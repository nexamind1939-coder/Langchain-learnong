from dotenv import load_dotenv

load_dotenv()

import os
from dotenv import load_dotenv

# Load keys from .env file
load_dotenv()

# Turn LangSmith tracking ON
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_PROJECT"] = "my-first-app"

# Check whether LangSmith key is available
if os.getenv("LANGSMITH_API_KEY"):
    print("LangSmith API key found!")
    print("LangSmith tracking is enabled!")
else:
    print("Please add your LANGSMITH_API_KEY to the .env file.")

import os
from dotenv import load_dotenv
from langsmith import traceable

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load LangSmith settings
load_dotenv()

# 1. LangChain chain tracing
prompt = ChatPromptTemplate.from_template(
    "Explain {topic} in one simple sentence."
)

model = ChatOllama(model="qwen2.5")

chain = prompt | model | StrOutputParser()

print("Running LangChain chain...")
print(chain.invoke({"topic": "LangSmith"}))


# 2. Custom Python function tracing
@traceable
def make_greeting(name):
    return f"Hello {name}, welcome to GenAI!"

print("\nRunning custom function...")
print(make_greeting("Shashi"))
