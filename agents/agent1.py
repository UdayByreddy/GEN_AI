from dotenv import load_dotenv
load_dotenv()
import os
import requests

from langchain_mistralai import ChatMistralAI
from langchain.tools import tool
from langchain_core.messages import HumanMessage,ToolMessage
from tavily import TavilyClient
from rich import print

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
    
    
    
llm = ChatMistralAI(
    model="mistral-small-2506",
)

tools = {
    "get_weather": get_weather,
    "get_news": get_news
}

llm_with_tool = llm.bind_tools([get_weather,get_news])

messages =[]

print("agent get news and weather of that city")

while True:
    user_input = input("You: ")
    if user_input=="exit":
        break
    messages.append(HumanMessage(content=user_input))
    while True:
        result = llm_with_tool.invoke(messages)
        messages.append(result)
        if result.tool_calls:
            for tool_call in result.tool_calls:
                tool_name = tool_call["name"]
                confirm = input(f"Do you want to use {tool_name} tool? (yes/no): ")
                if confirm.lower() == "no":
                    print("Skipping tool")
                    break;
                tool_result = tools[tool_name].invoke(tool_call)
                messages.append(ToolMessage(content=tool_result,
                                            tool_call_id=tool_call["id"]))
                
            continue
        else:
            print(result.content)
            break
                
                    
                
            
    
    

