from dotenv import load_dotenv
load_dotenv()
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel,RunnablePassthrough


model = ChatMistralAI(
    model="mistral-small-2506",
    temperature=0.9
)

prompt1 = ChatPromptTemplate.from_messages([
    ('system',"""
     you are ai coding agent that explain the code given to you""")
    ,
    ('human',"""
    {topic}
    """)
])

prompt2 = ChatPromptTemplate.from_messages([
    ('system',"""
     you are ai coding agent that explain the code given to you""")
    ,
    ('human',"""
    {code}
    """)
])

parser = StrOutputParser()

chain1 = prompt1 | model | parser 

chain2 = RunnableParallel({
    "code": RunnablePassthrough(),
    "detail": prompt2 | model | parser
})
chain = chain1 | chain2

output = chain.invoke({
    "topic":"write the code for binary search",
})

print(output.get("code"))
print(output.get("detail"))
 
