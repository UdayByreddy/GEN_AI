from langchain_community.document_loaders import WebBaseLoader

url = WebBaseLoader("https://www.apple.com/in/iphone-17/")

docs =  url.load()

print(docs[0].page_content)

print(len(docs))