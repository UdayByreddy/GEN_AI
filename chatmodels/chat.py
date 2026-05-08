from dotenv import load_dotenv

load_dotenv()

# from langchain_google_genai import ChatGoogleGenerativeAI

from langchain.chat_models import init_chat_model

from langchain_mistralai import ChatMistralAI

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint



# model = ChatGoogleGenerativeAI(
#     model="gemini-2.5-flash-lite"
# )

# model = init_chat_model("groq:openai/gpt-oss-120b")

model = ChatMistralAI(
    model= "mistral-small-2506",
    temperature=0,
    max_tokens=20
)

llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-R1"
)

model1 = ChatHuggingFace(llm=llm)


response = model1.invoke("what is the meaning of life in one line?")
print(response.content)