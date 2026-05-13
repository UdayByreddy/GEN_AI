from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_mistralai import ChatMistralAI
docs = [
    Document(page_content="Gradient descent is an optimization algorithm used in machine learning."),
    Document(page_content="Gradient descent minimizes the loss function."),
    Document(page_content="Gradient descent is an optimization that minimizes the loss function."),
    Document(page_content="Neural networks use gradient descent for training."),
    Document(page_content="Support Vector Machines are supervised learning algorithms.")
]

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
)


vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=embedding_model
)
retriever = vectorstore.as_retriever()

llm = ChatMistralAI(
    model = "mistral-small-2506",
    temperature=0.9)

mutli_query = MultiQueryRetriever.from_llm(
    retriever= retriever,
    llm=llm
)

query = "what is gradient descent"

docs = mutli_query.invoke(query)

for doc in docs:
    print(doc.page_content) 
