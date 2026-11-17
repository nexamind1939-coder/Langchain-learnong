
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 1. Load local Ollama model
model = ChatOllama(model="qwen2.5")

# 2. Create output parser
output_parser = StrOutputParser()

# 3. Create prompt
prompt = ChatPromptTemplate.from_template(
    "Explain {topic} in one simple sentence."
)

# 4. Create chain
chain = prompt | model | output_parser

print("Chain is ready!")

# 5. Get answer
answer = chain.invoke({"topic": "Python"})

print("\nAnswer:")
print(answer)

# 6. Create a second prompt
tip_prompt = ChatPromptTemplate.from_template(
    "Give one short, friendly tip for a beginner learning {skill}."
)

tip_chain = tip_prompt | model | output_parser

print("\nBeginner Tip:")
print(tip_chain.invoke({"skill": "Python"}))
