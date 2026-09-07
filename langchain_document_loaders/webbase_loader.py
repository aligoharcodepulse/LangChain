from langchain_community.document_loaders import WebBaseLoader
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

url = "https://namal.edu.pk/founder"
loader = WebBaseLoader(url)
docs = loader.load()
# print(len(docs))
# print(docs[0].page_content)

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
    temperature=0.7,
    max_new_tokens=200
)

model = ChatHuggingFace(llm=llm)

parser = StrOutputParser()

prompt = PromptTemplate(
    template="Answer the following question \n {question} from the following text. \n {text}",
    input_variables=["question", "text"]
)

chain = prompt | model | parser
result = chain.invoke({'question':'Who is the founder of Namal University?', 'text': docs[0].page_content})
print(result)