"""Comparison of standard similarity retrieval vs MultiQueryRetriever.

MultiQueryRetriever uses an LLM to rewrite one user query into multiple
query variations. Each variation retrieves documents, then results are merged
and deduplicated, which usually improves recall and reduces single-query bias.
Because retrieval happens for multiple rewrites, final results can exceed the
`k` value used in each individual retrieval call.
"""

import os
# Workaround for OpenMP duplicate runtime initialization seen with FAISS on some macOS setups.
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain.retrievers.multi_query import MultiQueryRetriever
from dotenv import load_dotenv

# Load environment variables such as API keys from .env.
load_dotenv()

# Relevant health & wellness documents
all_docs = [
    Document(page_content="Regular walking boosts heart health and can reduce symptoms of depression.", metadata={"source": "H1"}),
    Document(page_content="Consuming leafy greens and fruits helps detox the body and improve longevity.", metadata={"source": "H2"}),
    Document(page_content="Deep sleep is crucial for cellular repair and emotional regulation.", metadata={"source": "H3"}),
    Document(page_content="Mindfulness and controlled breathing lower cortisol and improve mental clarity.", metadata={"source": "H4"}),
    Document(page_content="Drinking sufficient water throughout the day helps maintain metabolism and energy.", metadata={"source": "H5"}),
    Document(page_content="The solar energy system in modern homes helps balance electricity demand.", metadata={"source": "I1"}),
    Document(page_content="Python balances readability with power, making it a popular system design language.", metadata={"source": "I2"}),
    Document(page_content="Photosynthesis enables plants to produce energy by converting sunlight.", metadata={"source": "I3"}),
    Document(page_content="The 2022 FIFA World Cup was held in Qatar and drew global energy and excitement.", metadata={"source": "I4"}),
    Document(page_content="Black holes bend spacetime and store immense gravitational energy.", metadata={"source": "I5"}),
]

# Embedding model to convert text into vector representations.
embedding_model = OpenAIEmbeddings()

# Build an in-memory FAISS vector store from sample documents.
vectorstore = FAISS.from_documents(
    documents=all_docs,
    embedding=embedding_model
)

# Baseline retriever: single-query similarity search.
similarity_retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}  # Return top 5 similar documents
)

# Multi-query retriever: generates multiple query rewrites through the LLM.
# Each rewrite retrieves up to k docs, then unique results are merged.
multi_query_retriever = MultiQueryRetriever.from_llm(
    retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
    llm=ChatOpenAI(model="gpt-4")
)


# User query.
query = "How to improve energy levels and maintain balance?"

# Run retrieval using both strategies for comparison.
similarity_results = similarity_retriever.invoke(query)

multi_query_results = multi_query_retriever.invoke(query)

# Print baseline similarity results.
for i, doc in enumerate(similarity_results):
    print(f"\n--- Result {i+1} ---")
    print(doc.page_content)


"""
The model got confused as you can see the first 4 results are relevant to the query but the 5th result is not related to improving energy levels and maintaining balance.

O/P:
--- Result 1 ---
Drinking sufficient water throughout the day helps maintain metabolism and energy.

--- Result 2 ---
Mindfulness and controlled breathing lower cortisol and improve mental clarity.

--- Result 3 ---
Regular walking boosts heart health and can reduce symptoms of depression.

--- Result 4 ---
Deep sleep is crucial for cellular repair and emotional regulation.

--- Result 5 ---
The solar energy system in modern homes helps balance electricity demand.
"""

print("*"*150)

# Print MultiQueryRetriever results.
for i, doc in enumerate(multi_query_results):
    print(f"\n--- Result {i+1} ---")
    print(doc.page_content)


"""
The multi-query retriever should ideally provide more focused results related to improving energy levels and maintaining balance.

--- Result 1 ---
Drinking sufficient water throughout the day helps maintain metabolism and energy.

--- Result 2 ---
The solar energy system in modern homes helps balance electricity demand.

--- Result 3 ---
Consuming leafy greens and fruits helps detox the body and improve longevity.

--- Result 4 ---
Deep sleep is crucial for cellular repair and emotional regulation.

--- Result 5 ---
Mindfulness and controlled breathing lower cortisol and improve mental clarity.

--- Result 6 ---
Regular walking boosts heart health and can reduce symptoms of depression.
"""
