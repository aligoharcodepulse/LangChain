from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

load_dotenv()

embedding = OpenAIEmbeddings(model='text-embedding-3-large', dimensions=300)

documents = [
    "Virat Kohli is an Indian Cricketer known for his aggressive batting and leadership.",
    "MS Dhoni is a former indian captain famous for his calm demeanor and finishing skills.",
    "Sachin Tendulker, also known as the 'God of Cricket', holds many batting records.",
    "Rohit Sharma is known for his elegant batting and record-breaking double centuries.",
    "Jasprit Bumrah is an indian fast bowler known for his unorthodox actions and yorkers."
]

query = 'Tell me about Virat Kohli'

document_embeddings = embedding.embed_documents(documents)
query_embedding = embedding.embed_query(query)

scores = cosine_similarity([query_embedding], document_embeddings)[0]
index, score = sorted(list(enumerate(scores)), key=lambda x:x[1])[-1]

print(query)
print(documents[index]) 
print("Similarity Score is:", score)