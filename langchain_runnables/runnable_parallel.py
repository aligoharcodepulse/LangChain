from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence, RunnableParallel

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
    temperature=0.7,
    max_new_tokens=200
)

model = ChatHuggingFace(llm=llm)

prompt1 = PromptTemplate(
    template="Generate a tweet about {topic}",
    input_variables=["topic"]
)

prompt2 = PromptTemplate(
    template="Generate a linkedin post about {topic}",
    input_variables=["topic"]
)

parser = StrOutputParser()

parallel_chain = RunnableParallel({
    'tweet':RunnableSequence(prompt1, model, parser),
    'linkedin':RunnableSequence(prompt2, model, parser)
})

result = parallel_chain.invoke({'topic':'google office in banglore'})
print(result)
# print(result['tweet'])
# print(result['linkedin'])