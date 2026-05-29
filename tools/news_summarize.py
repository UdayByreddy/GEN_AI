from dotenv import load_dotenv
load_dotenv()

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

search_tool = TavilySearchResults(
    max_results=5
)

llm = ChatGroq(
      model="llama-3.3-70b-versatile",
    temperature=0.9,
)

format_prompt = ChatPromptTemplate.from_messages([
    ('system',"""
     you are ai assistant that explain the news given to you into clear bullent points {news}"""),
    
])

output_parser = StrOutputParser()

chain =  format_prompt | llm | output_parser

result = search_tool.run("Latest news from andhra pradesh?")

news_result = chain.invoke({
    "news": result
})

print(news_result)