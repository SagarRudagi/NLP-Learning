"""Example of using RecursiveCharacterTextSplitter from LangChain.

This splitter is more intelligent than a plain character-based split because it
tries to break text at natural boundaries like spaces, sentences, or paragraphs
before reaching the chunk-size limit. This helps preserve meaning and readability
when splitting long documents for LLM input.
"""

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

# Basic text splitting example.
# This shows how a long paragraph can be split into smaller, more readable chunks.
text = "Machine learning models are increasingly being used to analyze large volumes of data and identify patterns that may be difficult for humans to spot manually. In healthcare, they can help doctors detect early warning signs, improve diagnosis, and recommend more personalized treatment plans. In business, these systems can support forecasting, automate repetitive tasks, and improve customer experiences through better personalization. However, machine learning is not without challenges: training models requires high-quality data, careful evaluation, and attention to fairness, privacy, and transparency. As the technology continues to evolve, it is important to use it responsibly and thoughtfully in real-world applications."

# RecursiveCharacterTextSplitter tries multiple separators in order, such as newlines,
# spaces, and punctuation, to keep chunk boundaries meaningful instead of chopping text arbitrarily.
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=50, # Maximum number of characters per chunk
    chunk_overlap=0, # Number of characters to overlap between chunks
)

# Split a raw string into multiple chunks.
text_chunks = text_splitter.split_text(text)
print(len(text_chunks))
print(text_chunks) # Print the generated chunks

# Load from Documents.
# This example loads a PDF and then splits the loaded Document objects into chunks.
loader = PyPDFLoader("gcpServices.pdf")

docs = loader.load()

doc_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100, # Maximum number of characters per chunk
    chunk_overlap=0, # Number of characters to overlap between chunks
)

# Split the loaded PDF documents into smaller, semantically cleaner chunks.
chunks = doc_splitter.split_documents(docs)

print(len(chunks))
print(chunks[0].page_content) # Print the content of the first split chunk


