from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader('Assignment1.pdf')
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size = 300,
    chunk_overlap = 0
)

chunks = splitter.split_documents(docs)
print(len(chunks))
print(chunks[1].page_content)
