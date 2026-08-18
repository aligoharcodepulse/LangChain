from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V4-Flash-0731",
    task="text-generation",
    max_new_tokens=200
)

model = ChatHuggingFace(llm=llm)

prompt = PromptTemplate(
    template="Generate five interesting facts about {topic}",
    input_variables=["topic"]
)

parser = StrOutputParser()

chain = prompt | model | parser

result = chain.invoke({
    "topic": "football"
})

print(result)

print("\n========== GRAPH ==========")
chain.get_graph().print_ascii()


# result = model.invoke("Generate five interesting facts about cricket")
# print(result)