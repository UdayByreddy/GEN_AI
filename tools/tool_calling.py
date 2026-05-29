from dotenv import load_dotenv
load_dotenv()

from langchain_mistralai import ChatMistralAI
from langchain.tools import tool
from rich import print
from langchain.messages import HumanMessage

@tool
def get_text_length(text:str) -> int:
    """ Calculate the length of the given text and return it as an integer."""
    return len(text)


tools = {
    "get_text_length": get_text_length
}
llm = ChatMistralAI(
    model="mistral-small-2506",
    temperature=0.9
)
#tool binding

llm_with_tool = llm.bind_tools([get_text_length])

message =[]

query = HumanMessage("Calculate the length of the given text and return it as an integer:'Hello are you inside the length'")

message.append(query)

print(message)
result = llm_with_tool.invoke(message)

message.append(result)

if result.tool_calls:
    tool_name = result.tool_calls[0]['name']
    result1 = tools[tool_name].invoke(result.tool_calls[0])
    message.append(result1)
    print(result1)
    
result2 = llm_with_tool.invoke(message)
print(result2.content)



# result = llm.invoke("Hello");
# print(result)
# print()
# print()
# result1 = llm_with_tool.invoke(" Calculate the length of the given text and return it as an integer: 'Hello are you inside the length' ")
# print(result1)

# if result1.tool_calls:
#     tool_calls = result1.tool_calls[0]

# tool_name = tool_calls['name']
# tool_args = tool_calls['args']

# result = get_text_length.invoke(tool_args)
# final_reponse = llm_with_tool.invoke(f"The length of the given text is: {result}")

# print(final_reponse)