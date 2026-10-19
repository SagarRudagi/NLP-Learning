"""Maximal Marginal Relevance (MMR) retriever example.

MMR retrieval balances two goals: relevance to the query and diversity among the
returned results. Instead of returning only near-duplicate top matches, it picks
documents that are relevant but also different from each other, which usually
gives broader context for RAG and question answering.
"""

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS # Just a different vector store

# Embedding model converts each document and query into vectors.
embedding_model = OpenAIEmbeddings()


# Sample knowledge-base documents for retrieval.
docs = [
    Document(page_content="LangChain makes it easy to work with LLMs."),
    Document(page_content="LangChain is used to build LLM based applications."),
    Document(page_content="Chroma is used to store and search document embeddings."),
    Document(page_content="Embeddings are vector representations of text."),
    Document(page_content="MMR helps you get diverse results when doing similarity search."),
    Document(page_content="LangChain supports Chroma, FAISS, Pinecone, and more."),
]

vectorstore = FAISS.from_documents(
    documents=docs,
    embedding=embedding_model
)

# Create a retriever that uses MMR (maximal marginal relevance).
# k=2: return top 2 final documents.
# lambda_mult controls relevance-vs-diversity balance.
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 2, "lambda_mult": 0.5}  # lambda_mult controls the trade-off between relevance and diversity. If the value is 1, it prioritizes relevance fully and if the value is 0, it prioritizes diversity fully.
)

# Query the retriever.
query = "What is langchain?"
results = retriever.invoke(query)

# Print selected documents.
for i, doc in enumerate(results):
    print(f"\n--- Result {i+1} ---")
    print(doc.page_content)
