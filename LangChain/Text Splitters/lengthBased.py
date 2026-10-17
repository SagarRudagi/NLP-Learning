"""Example of using CharacterTextSplitter from LangChain to split large text into smaller chunks.

This splitter is useful when the input text is too long for a model context window
or when you want to process content in smaller, manageable pieces. It breaks text
using character count, which makes it easy to control chunk size and keep each
segment readable and consistent.
"""

from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

# Basic text splitting example.
# This shows how a long paragraph can be split into smaller character-based chunks.
text = "Machine learning models are increasingly being used to analyze large volumes of data and identify patterns that may be difficult for humans to spot manually. In healthcare, they can help doctors detect early warning signs, improve diagnosis, and recommend more personalized treatment plans. In business, these systems can support forecasting, automate repetitive tasks, and improve customer experiences through better personalization. However, machine learning is not without challenges: training models requires high-quality data, careful evaluation, and attention to fairness, privacy, and transparency. As the technology continues to evolve, it is important to use it responsibly and thoughtfully in real-world applications."

# CharacterTextSplitter splits text by characters, not by semantic sentences.
# chunk_size controls the maximum size of each chunk, and separator=" " splits on spaces.
text_splitter = CharacterTextSplitter(
    chunk_size=50, # Maximum number of characters per chunk
    chunk_overlap=0, # Number of characters to overlap between chunks
    separator=" " # Character to use for splitting text
)

# Split a raw string into multiple chunks.
text_chunks = text_splitter.split_text(text)
print(len(text_chunks))
print(text_chunks[0]) # Print the first text chunk

# Load from Documents.
# This example loads a PDF and then splits the loaded Document objects into chunks.
loader = PyPDFLoader("gcpServices.pdf")

docs = loader.load()

doc_splitter = CharacterTextSplitter(
    chunk_size=100, # Maximum number of characters per chunk
    chunk_overlap=0, # Number of characters to overlap between chunks
    separator=" " # Character to use for splitting text
)

# Split Document objects into multiple smaller documents/chunks.
chunks = doc_splitter.split_documents(docs)

print(len(chunks))
print(chunks[0].page_content) # Print the content of the first chunk


