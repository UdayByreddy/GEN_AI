from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embeddings = OpenAIEmbeddings(
    model = 'text-embedding-3-large',
    dimensions=64
)
texts = [
    "What is life in one sentence?",
    "What is the meaning of life in one sentence?"
    "The quick brown fox jumps over the lazy dog"
]

vector = embeddings.embed_query("What is life in one sentence?")

vector = embeddings.embed_documents(texts)

print(len(vector))
print(vector)