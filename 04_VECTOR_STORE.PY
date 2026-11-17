
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# 1. Load local embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# 2. Create sample texts
texts = [
    "Python is a programming language.",
    "LangChain helps build AI applications.",
    "Streamlit creates web applications.",
    "FAISS stores and searches vector embeddings."
]

# 3. Create FAISS vector database
vectorstore = FAISS.from_texts(
    texts,
    embedding=embeddings
)

print("FAISS database created successfully!")

# 4. Search for similar text
query = "How can I build AI applications?"

results = vectorstore.similarity_search(query, k=2)

print("\nSearch Results:")

for result in results:
    print(result.page_content)

# ==========================================
# LANGCHAIN + HUGGING FACE + FAISS
# Vector Store and Similarity Search
# ==========================================

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# STEP 1: Load Hugging Face embedding model

print("Loading Hugging Face embedding model...")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("Embedding model loaded successfully!")


# STEP 2: Create sample text documents

texts = [
    "Dogs are friendly domestic animals and make wonderful pets.",
    "Cats are popular household pets. They enjoy sleeping and playing.",
    "Birds are beautiful animals. Some birds can learn to speak.",
    "Fish live in water and are commonly kept in aquariums.",
    "Python is a popular programming language.",
    "LangChain helps developers build AI applications.",
    "Streamlit is used to create interactive web applications."
]

print("\nTotal documents:", len(texts))


# STEP 3: Create FAISS vector store

print("\nCreating FAISS vector database...")

vector_store = FAISS.from_texts(
    texts,
    embedding=embeddings
)

print("FAISS vector database created successfully!")


# STEP 4: Search for similar documents

query = "How do I build websites?"

results = vector_store.similarity_search(
    query,
    k=2
)

print("\nTop matches for:", query)

for r in results:
    print("-", r.page_content)


# STEP 5: Try another search

query2 = "Tell me about programming"

results2 = vector_store.similarity_search(
    query2,
    k=2
)

print("\nTop matches for:", query2)

for r in results2:
    print("-", r.page_content)


# STEP 6: Save FAISS database locally

vector_store.save_local("faiss_index")

print("\nFAISS database saved successfully!")

print("\nProgram completed successfully!")
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Load embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Load saved FAISS database
vector_store = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)

print("FAISS database loaded successfully!")

# Search
query = "Tell me about pets"

results = vector_store.similarity_search(query, k=2)

print("\nSearch Results:")

for result in results:
    print("-", result.page_content)