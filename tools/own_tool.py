from langchain.tools import tool

@tool
def get_greetings(name: str) -> str:
    """Generate a greeting message to user"""
    return f"Hello {name}, welcome to LangChain!"


result = get_greetings.invoke({
    "name": "Uday"
})

print(result)
print(get_greetings.description)
print(get_greetings.args)