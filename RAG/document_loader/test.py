from langchain_community.document_loaders import TextLoader

from langchain_text_splitters import CharacterTextSplitter

splitter = CharacterTextSplitter(
    separator="",
    chunk_size=10,
    chunk_overlap=1
)

data = TextLoader("RAG/document_loader/notes.txt")



docs = data.load()

chucks =splitter.split_documents(docs)

for i in chucks:
    print(i.page_content)
    print(" -------------------> ")