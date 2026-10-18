"""Chroma vector store example with document CRUD and semantic search.

This script demonstrates the full lifecycle of vector-store usage in LangChain:
creating documents, searching by similarity, filtering with metadata, updating
an existing record, deleting a record, and fetching stored content for review.
"""

from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.schema import Document
from dotenv import load_dotenv

# Load environment variables such as API keys from a .env file.
load_dotenv()


# Create sample LangChain Document objects for IPL players.

doc1 = Document(
        page_content="Virat Kohli is one of the most successful and consistent batsmen in IPL history. Known for his aggressive batting style and fitness, he has led the Royal Challengers Bangalore in multiple seasons.",
        metadata={"team": "Royal Challengers Bangalore"}
    )
doc2 = Document(
        page_content="Rohit Sharma is the most successful captain in IPL history, leading Mumbai Indians to five titles. He's known for his calm demeanor and ability to play big innings under pressure.",
        metadata={"team": "Mumbai Indians"}
    )
doc3 = Document(
        page_content="MS Dhoni, famously known as Captain Cool, has led Chennai Super Kings to multiple IPL titles. His finishing skills, wicketkeeping, and leadership are legendary.",
        metadata={"team": "Chennai Super Kings"}
    )
doc4 = Document(
        page_content="Jasprit Bumrah is considered one of the best fast bowlers in T20 cricket. Playing for Mumbai Indians, he is known for his yorkers and death-over expertise.",
        metadata={"team": "Mumbai Indians"}
    )
doc5 = Document(
        page_content="Ravindra Jadeja is a dynamic all-rounder who contributes with both bat and ball. Representing Chennai Super Kings, his quick fielding and match-winning performances make him a key player.",
        metadata={"team": "Chennai Super Kings"}
    )

docs = [doc1, doc2, doc3, doc4, doc5]

# Initialize Chroma with OpenAI embeddings.
# - persist_directory stores the database on disk.
# - collection_name identifies a logical dataset within Chroma.
vector_store = Chroma(
    embedding_function=OpenAIEmbeddings(),
    persist_directory="chromaDB",
    collection_name="sample"
)

# Add documents once to populate the collection (keep commented to avoid duplicates on rerun).
# vector_store.add_documents(docs)

# Fetch full stored payload for inspection (embeddings can be large).
response = vector_store.get(include=["embeddings", "documents", "metadatas"])

# print(response)

# Similarity search: returns top-k matching documents.
bowler = vector_store.similarity_search(
query = "Who among them is a bowler?",
k=2 # Number of similar documents to retrieve
)

# print(bowler)


# Similarity search with scores: lower score generally means closer match.
bowler_with_score = vector_store.similarity_search_with_score(
    query="Who among them is a bowler?",
    k=2
)

# print(bowler_with_score)


# Similarity search with metadata filter to narrow results to a specific team.
bowler_with_metadata = vector_store.similarity_search_with_score(
    query="Who among them is a bowler?",
    k=2,
    filter={"team": "Mumbai Indians"}
)

# print(bowler_with_metadata)

# Retrieve document payload to inspect IDs (IDs are returned by default).
# document_id = vector_store.get(include=["documents"])
# print(document_id)


# Update operation:
# Create a replacement Document, then update by the exact Chroma document ID.
updated_doc1 = Document(
    page_content="Virat Kohli, the former captain of Royal Challengers Bangalore (RCB), is renowned for his aggressive leadership and consistent batting performances. He holds the record for the most runs in IPL history, including multiple centuries in a single season. Despite RCB not winning an IPL title under his captaincy, Kohli's passion and fitness set a benchmark for the league. His ability to chase targets and anchor innings has made him one of the most dependable players in T20 cricket.",
    metadata={"team": "Royal Challengers Bangalore"}
)

vector_store.update_document(document_id='306d267f-5028-49ec-894e-17e9bd4088d2', document=updated_doc1)

# document_id = vector_store.get(include=["documents"])
# print(document_id)


# Delete operation by ID.
vector_store.delete(ids=['306d267f-5028-49ec-894e-17e9bd4088d2'])

# Verify the current collection state after deletion.
document_id = vector_store.get(include=["documents"])
print(document_id)