from langchain_community.retrievers import ArxivRetriever


retriever = ArxivRetriever(
    load_max_docs=2,
    load_all_available_meta=True
)

docs = retriever.invoke("artificial intelligence")

for i,doc in enumerate(docs):
    print(i)
    print(doc.page_content[:500])
    print(doc.metadata)