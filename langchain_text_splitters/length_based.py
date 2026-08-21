from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader('Assignment1.pdf')
docs = loader.load()

# text = """
# Artificial Intelligence (AI) is a field of computer science that focuses on creating machines and software that can perform tasks that normally require human intelligence. These tasks include understanding language, recognizing images, making decisions, solving problems, learning from data, and recognizing patterns.

# Machine learning is one of the most important areas of artificial intelligence. Instead of programming a computer with every possible rule, machine learning allows a system to learn patterns from data. For example, a machine learning model can be trained using thousands of images of cats and dogs. After training, the model can use the learned patterns to predict whether a new image contains a cat or a dog.

# Deep learning is a specialized area of machine learning that uses artificial neural networks with many layers. Deep learning models are particularly useful for complex tasks such as image recognition, speech recognition, natural language processing, and autonomous driving. Large amounts of data and powerful computing resources are often required to train these models.

# Natural Language Processing, commonly called NLP, allows computers to understand, process, and generate human language. NLP is used in applications such as chatbots, translation systems, search engines, voice assistants, and text summarization tools. Modern NLP systems can generate human-like responses by learning patterns from large collections of text.

# """

splitter = CharacterTextSplitter(
    chunk_size = 300,
    chunk_overlap = 0,
    separator='\n\n'
)

chunks = splitter.split_documents(docs)
print(chunks[0].page_content)