from dotenv import load_dotenv

load_dotenv()

from langchain_mistralai import ChatMistralAI

from langchain_community.document_loaders import TextLoader,PyPDFLoader,WebBaseLoader


from langchain_core.prompts import ChatPromptTemplate

data = TextLoader("RAG/document_loader/notes.txt")

data1 = PyPDFLoader("RAG/document_loader/udaykiran-resume.pdf")


docs = data.load()

docs1 = data1.load()


template = ChatPromptTemplate.from_messages([
    ("system",
     "Please review this PR"
    
),("human","{data}")])


prompt = template.format_messages(data = docs[0].page_content)

model = ChatMistralAI(model="mistral-small"
                      ,temperature=0.9)
response = model.invoke(prompt)

print(response.content)
