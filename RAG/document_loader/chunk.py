from langchain_community.document_loaders import PyPDFLoader
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
load_dotenv()


data = PyPDFLoader("RAG/document_loader/deep_learning.pdf")

docs = data.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

prompt = ChatPromptTemplate.from_messages([(
    'system',
    'You are AI that summanry the text given to you'),
    ('human', '{data}')
])

chunks = splitter.split_documents(docs)


model = ChatMistralAI(
    model="mistral-small-2506",
    temperature=0.9
)

final_prompt = prompt.format(data=docs)

response = model.invoke(final_prompt)

print(response.content)