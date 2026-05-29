from dotenv import load_dotenv
load_dotenv()
import os
import requests

from langchain_mistralai import ChatMistralAI
from langchain.tools import tool
from langchain_core.messages import HumanMessage,ToolMessage
from tavily import TavilyClient
from rich import print
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_tool_call
@tool
def get_weather(city:str) -> str:
    """Get current weather of a city"""
    api_key = os.getenv("OPENWEATHER_API_KEY")
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city},IN&appid={api_key}&units=metric"
    response = requests.get(url)
    data = response.json()
    print(data)
    if response.status_code !=200:
        return f"Error: {data['message']}"
    
    temp = data["main"]['temp']
    desc = data["weather"][0]["description"]
    
    return f"The temperature in {city} is {temp} degrees Celsius with {desc}"


tavliy = TavilyClient(api_key = os.getenv("TAVILY_API_KEY"))

@tool
def get_news(city:str)->str:
    """Get latest news about the city"""
    query = f"Latest news from {city}"
    response = tavliy.search(
        query=query,
        max_results=5,
    )
    results = response.get("results",[]);
    print(results)
    if not results:
        return f"No news found for{city}"
    news =[]
    for r in results:
        title = r.get("title","")
        url = r.get("url","")
        snippet = r.get("snippet","")
        news.append(f"{title} - {url}- {snippet}")
    return "\n".join(news)
    
 
@wrap_tool_call
def human_approve(request,handler):
    """Ask the user if they want to use the tool"""
    tool_name = request.tool_call["name"]
    confirm = input(f"Do you want to use {tool_name} tool? (yes/no): ")
    if confirm.lower() != "yes":
        return ToolMessage(content="User declined to use tool",tool=tool_name)
    return handler(request)
        
       
    
llm = ChatMistralAI(
    model="mistral-small-2506",
)


agent = create_agent(llm,
                     tools=[get_weather,get_news],
                     system_prompt="you are a helpful city assitant",
                     middleware= [human_approve]
                     
        )

print("agent get news and weather of that city")

while True:
    user_input = input("You: ")
    if user_input.lower()=="exit":
        break
    result = agent.invoke(
        {"messages": [{"role": "user", "content": user_input}]})
    print(result['messages'][-1].content)
    
    

