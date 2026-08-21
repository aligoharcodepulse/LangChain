from langchain_text_splitters import RecursiveCharacterTextSplitter, Language

text = """
# Artificial Intelligence

Artificial Intelligence (AI) is a field of computer science that enables machines to perform tasks that normally require human intelligence.

## Machine Learning

Machine Learning (ML) allows computers to learn patterns from data and make predictions without being explicitly programmed for every task.

## Generative AI

Generative AI can create new content such as text, images, audio, and computer code. Large Language Models (LLMs) are commonly used for generating and understanding text.

## Applications

AI is used in healthcare, education, finance, robotics, and software development. It can help analyze data, automate tasks, and provide intelligent recommendations.

## Challenges

AI can produce incorrect information and may have problems related to bias, privacy, and security. Responsible development and human supervision are therefore important.
"""

splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.MARKDOWN,
    chunk_size = 350,
    chunk_overlap = 0
)

chunks = splitter.split_text(text)
print(len(chunks))
print(chunks[0])