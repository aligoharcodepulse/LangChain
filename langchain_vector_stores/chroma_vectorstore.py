from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

doc1 = Document(
    page_content="Babar Azam is a top-order batter and captain of Peshawar Zalmi in the PSL. He is known for consistent run-scoring and leadership.",
    metadata={"team":"Peshawar Zalmi"}
)

doc2 = Document(
    page_content="Shaheen Afridi is a left-arm fast bowler and captain of Lahore Qalandars. He leads the bowling attack and is known for taking early wickets.",
    metadata={"team":"Lahore Qalandars"}
)

doc3 = Document(
    page_content="Shadab Khan is an all-rounder and captain of Islamabad United. He contributes with both leg-spin bowling and aggressive middle-order batting.",
    metadata={"team":"Islamabad United"}
)

docs = [doc1, doc2, doc3]

vector_store = Chroma(
    embedding_function=embeddings,
    persist_directory='my_chroma_db',
    collection_name='sample'
)

store_documents = vector_store.add_documents(
    docs,
    ids=["babar", "shaheen", "shadab"]
)

print(store_documents)
retrieve_documents = vector_store.get(include=['embeddings', 'documents', 'metadatas'])
# print(retrieve_documents)

# similarity_check = vector_store.similarity_search(
#     query='Who among these are a bowler?',
#     k=1 # how much similar vectors you want to display
# )

# print(similarity_check)

# similarity_check1 = vector_store.similarity_search_with_score(
#     query='Who among these is a batsman?',
#     k=1 # how much similar vectors you want to display
# )

# print(similarity_check1)

# similarity_check2 = vector_store.similarity_search_with_score(
#     query='',
#     filter={'team':'Peshawar Zalmi'}
# )

# print(similarity_check2)


# update documents
updated_doc1 = Document(
    page_content="Babar Azam is a top-order battesman and captain of the Peshawar Zalmi in the PSL. He is known for consistency in run-scoring and leadership.",
    metadata={"team":"Peshawar Zalmi"}
)
print(vector_store.update_document(document_id='babar', document=updated_doc1))
retrieve_documents = vector_store.get(include=['embeddings', 'documents', 'metadatas'])
# print(retrieve_documents)


# Delete document
vector_store.delete(ids=['shadab'])
retrieve_documents = vector_store.get(include=['embeddings', 'documents', 'metadatas'])
print(retrieve_documents)