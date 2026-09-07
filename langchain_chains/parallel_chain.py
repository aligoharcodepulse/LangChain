from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableParallel

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V4-Flash-0731",
    task="text-generation",
    max_new_tokens=200
)

model1 = ChatHuggingFace(llm=llm)
model2 = ChatOpenAI()

prompt1 = PromptTemplate(
    template="Generate short and simple from the following. \n {text}",
    input_variables=["text"]
)

prompt2 = PromptTemplate(
    template="Generate 5 short question answers from the following text. \n {text}",
    input_variables=["text"]
)

prompt3 = PromptTemplate(
    template="Merge the provided notes and quiz into a single document. Notes -> {notes} and Quiz -> {quiz} \n {text}",
    input_variables=["notes","quiz"]
)

parser = StrOutputParser()

text = """
LangChain is an open-source framework designed to simplify the development of applications powered by large language models (LLMs). It provides tools and components that allow developers to connect language models with prompts, external data sources, APIs, databases, and other software systems.

One of the important features of LangChain is its prompt templates, which allow developers to create reusable prompts with dynamic inputs. LangChain also supports chains, where multiple operations can be connected together so that the output of one step becomes the input of another. For example, a developer can first ask an LLM to generate a detailed report and then pass that report to another prompt to create a short summary.

LangChain also provides output parsers that help convert raw model responses into structured formats. This makes it easier for applications to process information returned by an LLM. Developers can also use LangChain with different models and providers, including OpenAI, Hugging Face, and other LLM platforms.

Another major capability is retrieval-augmented generation (RAG). With RAG, LangChain can retrieve relevant information from documents or databases and provide that information to an LLM before generating an answer. This is useful for applications such as question-answering systems, document analysis, chatbots, and research assistants.

Because of these features, LangChain is widely useful for building AI applications, chatbots, research tools, document-processing systems, and intelligent automation systems. It provides a structured way to combine language models with other components required for real-world AI applications.

"""
parallel_chain = RunnableParallel(
    {
        'notes': prompt1 | model1 | parser,
        'quiz': prompt2 | model2 | parser
    }
)

merge_chain = prompt3 | model1 | parser

chain = parallel_chain | merge_chain
result = chain.invoke({'text': text})
# print(result)

chain.get_graph().print_ascii()