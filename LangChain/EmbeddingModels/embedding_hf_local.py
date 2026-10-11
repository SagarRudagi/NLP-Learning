"""
Brief explanation:
This script uses a local Hugging Face sentence-transformer model to create embeddings for
one text and a batch of texts.

Author's note:
This example helps show how to generate sentence embeddings without using an external API,
which is useful for offline or low-cost experimentation.
"""

from langchain_huggingface import HuggingFaceEmbeddings

# Initialize a local sentence-transformer model that turns text into vector embeddings.
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Embed a single sentence.
text = "Paris is the capital of France"
result = embeddings.embed_query(text)
print(str(result))

# Create embeddings for multiple sentences at once.
docs = [
    "Paris is the capital of France",
    "France is a country in Europe",
    "Europe is a continent",
    "The Earth is a planet in the Solar System"
]

results = embeddings.embed_documents(docs)
print(str(results))
