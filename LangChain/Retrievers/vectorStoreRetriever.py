"""Vector-store retriever example using Chroma and OpenAI embeddings.

This script converts plain text documents into embeddings, stores them in a
Chroma vector database, and retrieves the most relevant documents for a query.
It also compares retrieval through a retriever interface vs direct similarity
search on the vector store.
"""

from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()

# Step 1: Your source documents
documents = [
    Document(page_content="LangChain helps developers build LLM applications easily."),
    Document(page_content="Chroma is a vector database optimized for LLM-based search."),
    Document(page_content="Embeddings convert text into high-dimensional vectors."),
    Document(page_content="OpenAI provides powerful embedding models."),
]

# Step 2: Initialize embedding model
embedding_model = OpenAIEmbeddings()

# Step 3: Create Chroma vector store in memory and add documents to it
# from_documents() computes embeddings and stores them immediately.
vectorstore = Chroma.from_documents(
    documents=documents,
    embedding=embedding_model,
    collection_name="my_collection"
)

# Convert vector store into a retriever object.
# k=2 means return the top 2 most relevant documents.
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

# Query in natural language.
query = "What is Chroma used for?"

# Retrieval path 1: Use retriever abstraction.
results = retriever.invoke(query)

# Print retriever results.
for i, doc in enumerate(results):
    print(f"\n--- Result {i+1} ---")
    print(doc.page_content)


# Retrieval path 2: Call similarity_search directly on the vector store.
results = vectorstore.similarity_search(query, k=2)

# Print direct vector-store search results.
for i, doc in enumerate(results):
    print(f"\n--- Result {i+1} ---")
    print(doc.page_content)

# We use retriever instead of calling similarity_search directly because it provides a consistent interface for retrieval, which can be useful when switching between different vector stores or retrieval strategies.