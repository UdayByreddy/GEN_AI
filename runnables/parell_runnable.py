from langchain_mistralai import ChatMistralAI
from dotenv import load_dotenv  

load_dotenv()

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel,RunnableLambda


model = ChatMistralAI(
    model_name="mistral-small-2506",
    temperature=0.9,
)

short_prompt = ChatPromptTemplate.from_template("Expalin the {topic} in 1-2 lines")

long_prompt = ChatPromptTemplate.from_template("Expalin the {topic} in detail")

parse = StrOutputParser()


chain =RunnableParallel({
    "short": RunnableLambda(lambda x: x['short']) | short_prompt | model | parse,
    "long": RunnableLambda(lambda x: x['long']) |  long_prompt | model | parse
})

output = chain.invoke({"short":{
    "topic": "What is ai?"
},
"long":{
    "topic": "What is ml?"
}})

print(output.get('short'))
print('------>')
print(output.get('long'))