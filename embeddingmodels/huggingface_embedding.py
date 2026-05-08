from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()

embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
)

texts = [
    "What is life in one sentence?",
    "What is the meaning of life in one sentence?"
    "The quick brown fox jumps over the lazy dog"
]

vector = embedding.embed_documents(texts)

print(len(vector))
print(vector)