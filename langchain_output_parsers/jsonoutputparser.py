from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V4-Pro-0813",
    task="text-generation"
   # huggingfacehub_api_token=hf_token
)

model = ChatHuggingFace(llm=llm)

parser = JsonOutputParser()

template = PromptTemplate(
    template="Give me the name, age, and city of a fictional person. \n {format_instruction}",
    input_variables=[],
    partial_variables={'format_instruction':parser.get_format_instructions()}
)
# can write below three lines with the help of chains
# prompt = template.format()
# result = model.invoke(prompt)
# final_result = parser.parse(result.content)

# print(final_result)
# print(type(final_result))

# given as
chain = template | model | parser
result = chain.invoke({})
print(result)

# jsonoutputparser does not enforce schema (means if you want a result in your own format)
