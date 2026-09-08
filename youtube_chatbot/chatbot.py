from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, IpBlocked
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings, ChatHuggingFace, HuggingFaceEndpoint
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

# Step - 1 (Indexing)
video_id = "LPZh9BOjkQs"  # only the ID, not full URL

try:
    # If you don't care which language, this returns the "best" one
    ytt_api = YouTubeTranscriptApi()

    transcript = ytt_api.fetch(
        video_id,
        languages=["en"]
    )

    # Flatten it to plain text
    transcript = " ".join(chunk.text for chunk in transcript)
    #print(transcript)

except TranscriptsDisabled:
    print("No captions available for this video.")

except IpBlocked:
    print("YouTube is blocking your IP address.")
    print("Try another network or use a residential proxy.")

# Splitter
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.create_documents([transcript])
#print(len(chunks))

# Vextor Store
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(chunks, embeddings)
#print(vectorstore.index_to_docstore_id)


# Step - 2 (Retrieval)
retriever = vectorstore.as_retriever(search_type='similarity',search_kwargs={"k":4})
# print(retriever)
query = retriever.invoke("What is large language model?")
#print(query)


# Step - 3 (Augmentation)
llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
    max_new_tokens=512,
    temperature=0.2
)

model = ChatHuggingFace(llm=llm)

prompt = PromptTemplate(
    template="""
    You are a helpful assistant.
    Answer ONLY from the provided transcript context.
    If the context is insufficient, just say you don't know.

    {context}

    Question: {question}
    """,
    input_variables=["context", "question"]
)


question = "Is the topic of llm discussed in this video? if yes then what was discussed?"
retrieved_docs = retriever.invoke(question)
#print(retrieved_docs)

context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
#print(context_text)
final_prompt = prompt.invoke({"context":context_text, "question":question})


# Step - 4 (Generation)
answer = model.invoke(final_prompt)
print(answer)


