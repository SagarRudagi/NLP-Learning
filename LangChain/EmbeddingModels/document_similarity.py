"""
Brief explanation:
This script uses OpenAI embeddings to convert a query and several documents into vectors,
then finds which document is most semantically similar using cosine similarity.

Author's note:
This is a practical example of document retrieval and similarity matching in an embedding-based workflow.
"""

from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity

# Load the API credentials for the embedding model.
load_dotenv()

# Create an embedding model with a specific dimensionality.
embedding = OpenAIEmbeddings(model="text-embedding-3-large", dimensions=300)

# A small set of documents to compare against a user query.
docs = [
    "In Breaking Bad, Bryan Cranston plays Walter White, whose transformation drives the entire series.",
    "Stranger Things features Millie Bobby Brown as Eleven, a character with mysterious powers tied to the Upside Down.",
    "In Friends, Jennifer Aniston plays Rachel Green, whose on-and-off story with Ross became iconic.",
    "The Crown stars Olivia Colman as Queen Elizabeth II, portraying the pressure of royal duty.",
    "In The Mandalorian, Pedro Pascal plays Din Djarin, the bounty hunter protecting Grogu across the galaxy.",
]

# Define a query that should match one of the documents most closely.
query = "Which series portrays a dark plot with monsters?"

# Convert all documents and the query into embedding vectors.
doc_embeddings = embedding.embed_documents(docs)
query_embedding = embedding.embed_query(query)

# Compute similarity scores and select the highest-scoring document.
similarity_matrix = cosine_similarity([query_embedding], doc_embeddings)[0]
index, score = sorted(list(enumerate(similarity_matrix)), key=lambda x: x[1])[-1]

print(f"Query: {query}")
print(f"Doc: {docs[index]}")
print(f"Score: {score}")

