from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma


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
similarity_search = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)
result1 = similarity_search.invoke("Gradient descent")
 
for doc in result1:
    print(doc.page_content)

mmr_search = vectorstore.as_retriever(
   search_type="mmr",
   search_kwargs={"k": 3}
)

result2 = mmr_search.invoke("Gradient descent")

for doc in result2:
    print(doc.page_content)
 