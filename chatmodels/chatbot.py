from dotenv import load_dotenv
import os

from langchain_mistralai  import ChatMistralAI
from langchain.messages import AIMessage,SystemMessage,HumanMessage

load_dotenv()

print(os.getenv("MISTRAL_API_KEY"))
model = ChatMistralAI(
    model="mistral-small-2506",
    temperature=0.5
)
messages = [
    SystemMessage(content="You are a funny AI agent")
]
print("-----> Type 0 to exit ------>")
while True:
    prompt = input("You: ")
    messages.append(HumanMessage(content=prompt))
    if(prompt == "0"):
        break
    reponse = model.invoke(messages)
    messages.append(AIMessage(content=reponse.content))
    print("Bot: " + reponse.content)

