from dotenv import load_dotenv
load_dotenv()

from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

prompt = ChatPromptTemplate.from_template(
    "You are ai that expalin the answer:{topic}"
)

model = ChatMistralAI(
    model="mistral-small-2506",
)

parser = StrOutputParser()

# final_prompt = prompt.from_template(topic="What is ai?")

# reponse = model.invoke(final_prompt)


# output = parser.parse(reponse.content)

# print(output)

chain = prompt | model | parser

rep = chain.invoke("What is ai?")

print(rep)