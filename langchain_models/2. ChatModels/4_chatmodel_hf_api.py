from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import os

load_dotenv()

#hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V4-Flash-0731",
    task="text-generation"
   # huggingfacehub_api_token=hf_token
)

model = ChatHuggingFace(llm=llm)

result = model.invoke("Who is the prime minister of Pakistan?")

print(result.content)