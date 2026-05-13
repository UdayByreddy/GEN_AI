from langchain_community.document_loaders import PyPDFLoader

# from langchain_text_splitters import TokenTextSplitter

from langchain_text_splitters import RecursiveCharacterTextSplitter

data = PyPDFLoader("RAG/document_loader/deep_learning.pdf")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=50,
    chunk_overlap=10
)

docs = data.load()

chunks = splitter.split_documents(docs);


print(chunks)

for i in chunks:
    print(i.page_content)
    print(" -------------------> ")

print(len(chunks))