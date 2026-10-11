from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

load_dotenv()

embedding = OpenAIEmbeddings(model="text-embedding-3-large", dimensions=300)
docs = [
    "In Breaking Bad, Bryan Cranston plays Walter White, whose transformation drives the entire series.",
    "Stranger Things features Millie Bobby Brown as Eleven, a character with mysterious powers tied to the Upside Down.",
    "In Friends, Jennifer Aniston plays Rachel Green, whose on-and-off story with Ross became iconic.",
    "The Crown stars Olivia Colman as Queen Elizabeth II, portraying the pressure of royal duty.",
    "In The Mandalorian, Pedro Pascal plays Din Djarin, the bounty hunter protecting Grogu across the galaxy.",
]



query = "Which series portrays a dark plot with monsters?"

doc_embeddings = embedding.embed_documents(docs)

query_embedding = embedding.embed_query(query)


similarity_matrix = cosine_similarity([query_embedding], doc_embeddings)[0]
index,score = sorted(list(enumerate(similarity_matrix)), key=lambda x: x[1])[-1]

print(f"Query: {query}")
print(f"Doc: {docs[index]}")
print(f"Score: {score}")
