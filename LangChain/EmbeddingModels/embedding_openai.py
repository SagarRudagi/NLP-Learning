"""
Brief explanation:
This script demonstrates how to generate embeddings for a single sentence and for multiple
sentences using OpenAI's embedding model in LangChain.

Author's note:
The script is intended as a quick hands-on example of how vector representations are created
for semantic search and downstream similarity tasks.
"""

from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

# Load environment variables needed for OpenAI access.
load_dotenv()

# Create an embedding model; a smaller vector size is used here for readability in examples.
embedding = OpenAIEmbeddings(model="text-embedding-3-large", dimensions=32)

# Embed a single sentence and print its vector representation.
result = embedding.embed_query("Paris is the capital of France")
print(str(result))
# Output: [-0.1661376953125, 0.49267578125, ...]

# Generate embeddings for several documents so they can later be compared semantically.
docs = [
    "Paris is the capital of France",
    "France is a country in Europe",
    "Europe is a continent",
    "The Earth is a planet in the Solar System"
]

results = embedding.embed_documents(docs)
print(str(results))
# Output: A list of embedding vectors, one per document.
