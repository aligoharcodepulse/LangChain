from langchain_core.tools import tool
import requests
from langchain_core.messages import HumanMessage
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()

# tool create
@tool
def multiply(a:int, b:int)->int:
    """Given two number a and b this tool return their product"""
    return a * b

print(multiply.invoke({'a':3, 'b':4}))
print(multiply.name)
print(multiply.description)
print(multiply.args)


# tool binding
llm = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    task="text-generation",
    max_new_tokens=100,
    temperature=0.7,
)

model = ChatHuggingFace(llm=llm)

llm_with_tools = model.bind_tools([multiply])
print(llm_with_tools)
