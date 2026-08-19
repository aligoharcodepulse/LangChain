from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    task="text-generation",
    max_new_tokens=200,
    temperature=0.7
)

model = ChatHuggingFace(llm=llm)

prompt = PromptTemplate(
    template="Suggest a catchy blog title about {topic}",
    input_variables=["topic"]
)

topic = input("Enter a topic: ")

# We don't need to format the prompt etc 
chain = prompt | model
result = chain.invoke(topic)
print("Generated blog title:", result.content)
