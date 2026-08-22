from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import LLMChainExtractor
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
    max_new_tokens=200,
    temperature=0.7
)

model = ChatHuggingFace(llm=llm)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# Recreate the document objects from the previous data

docs = [
    Document(
        page_content=(
            """The Grand Canyon is one of the most visited natural wonders in the world.
                Photosynthesis is the process by which green plants convert sunlight into energy.
                Millions of tourists travel to see it every year. The rocks date back millions of years."""
        ),
        metadata={"source": "Doc1"}
    ),

    Document(
        page_content=(
            """In medieval Europe, castles were built primarily for defense.
                The chlorophyll in plant cells captures sunlight during photosynthesis.
                Knights wore armor made of metal. Siege weapons were often used to breach castle walls."""
        ),
        metadata={"source": "Doc2"}
    ),

    Document(
        page_content=(
            """Basketball was invented by Dr. James Naismith in the late 19th century.
                It was originally played with a soccer ball and peach baskets. 
                NBA is now a global league."""
        ),
        metadata={"source": "Doc3"}
    ),

    Document(
        page_content=(
            """The history of cinema began in the late 1800s. Silent films were the earliest form.
                Thomas Edison was among the pioneers. Photosynthesis does not occur in animal cells.
                Modern filmmaking involves complex CGI and sound design"""
        ),
        metadata={"source": "Doc4"}
    )
]

vectorstore = FAISS.from_documents(docs, embeddings)
base_retriever = vectorstore.as_retriever(
    search_kwargs={"k": 4}
)
compressor = LLMChainExtractor.from_llm(model)

compression_retriever = ContextualCompressionRetriever(
    base_retriever=base_retriever,
    base_compressor=compressor
)

query = "What is photosynthesis?"

results = compression_retriever.invoke(query)

for i, doc in enumerate(results):
    print(f"\n--- Result {i+1} ---")
    print(doc.page_content)
    print(doc.metadata)

