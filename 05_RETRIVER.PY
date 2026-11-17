
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# 1. Sample documents
texts = [
    "Python is a programming language used for AI.",
    "Streamlit is used to build web apps quickly.",
    "LangChain helps connect AI models with your data.",
    "Dogs are friendly and loyal animals."
]

# 2. Load the embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# 3. Create FAISS vector store
vector_store = FAISS.from_texts(texts, embeddings)

print("Vector store ready!")

# 4. Create retriever (top 2 results)
retriever = vector_store.as_retriever(
    search_kwargs={"k": 2}
)

print("Retriever is ready!")

# 5. Ask a question
question = "What is used to build web apps?"

# 6. Retrieve relevant documents
results = retriever.invoke(question)

print("Question:", question)
print("\nRelevant information found:")

for r in results:
    print("-", r.page_content)
