from langchain_huggingface import HuggingFaceEmbeddings

print("Loading embedding model...")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("Embedding model loaded!")

vector = embeddings.embed_query("LangChain helps build AI applications.")

print("Embedding successful!")
print("Vector dimensions:", len(vector))
print("First 5 values:", vector[:5])