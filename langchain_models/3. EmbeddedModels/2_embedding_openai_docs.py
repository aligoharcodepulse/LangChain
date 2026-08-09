from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embedding = OpenAIEmbeddings(model='text-embedding-3-large', dimensions=32)

docs = [
    "Islamabad is the capital of Pakistan", 
    "Istanbul is the capital of Turkiye",
    "Delhi is the capital of India"
]
result = embedding.embed_documents(docs)
print(str(result))