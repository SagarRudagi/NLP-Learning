"""BM25 retriever example using keyword-based search.

BM25 is a sparse retrieval algorithm used in classical information retrieval.
It scores each document by combining:
1) term frequency (how often query words appear in the document),
2) inverse document frequency (how informative/rare those words are in the corpus),
3) and document-length normalization (to avoid unfairly favoring long documents).

Unlike embedding-based retrievers, BM25 does not understand semantic meaning;
it relies on lexical overlap between query and document text. This makes it
fast, interpretable, and strong for exact-term search, technical keywords, and
domains where wording matters. A common pattern in RAG systems is to use BM25
as a first-pass retriever or combine it with dense retrieval for better recall.
"""

from langchain_community.retrievers import BM25Retriever
from langchain.schema import Document

# Sample corpus
# BM25 works best with text where exact keyword matching matters.
docs = [
    Document(page_content="BM25 is a ranking function used for information retrieval."),
    Document(page_content="BM25 considers term frequency and document length."),
    Document(page_content="Dense retrievers use embeddings to find semantically similar documents."),
    Document(page_content="Retrieval is a critical step in building a chatbot or question-answering system."),
    Document(page_content="Sparse retrievers like BM25 are keyword-based and interpretable.")
]

# Create a BM25 retriever and load documents
# from_documents() builds an internal sparse index from the provided corpus.
retriever = BM25Retriever.from_documents(docs)

# Set how many top documents to return
# k controls the number of highest-scoring matches returned for each query.
retriever.k = 3

# Query example
# BM25 compares query terms against document terms using lexical relevance.
query = "how does bm25 work for retrieval?"

# Retrieve relevant documents
# invoke() is the Runnable-style API for calling retrievers in modern LangChain.
results = retriever.invoke(query)

# Display the results
print("Top Retrieved Documents:\n")
for i, doc in enumerate(results, 1):
    print(f"{i}. {doc.page_content}\n")