"""DirectoryLoader example for loading multiple PDF files from a folder.
Problem: This approach loads all PDF files in the specified directory into memory at once, which may not be efficient for very large collections of PDFs. This problem is resolved by lazy loading.

This script demonstrates how all PDF files in a directory can be loaded into
LangChain documents at once, making it easy to process large batches of files.

Difference between load() and lazy_load():

| Method      | Behavior                            | Memory usage | Best use case                          |
|-------------|------------------------------------|--------------|----------------------------------------|
| load()      | Loads all documents immediately     | High         | Small datasets or when you need all    |
| lazy_load() | Yields documents one at a time      | Low          | Large datasets, streaming, memory-safe, chunking, embedding |

load() is eager and keeps everything in memory, while lazy_load() is lazy and
processes content as needed. For big document collections, lazy_load() is more
memory efficient and usually a better choice.
"""

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
# Load all files from a directory into LangChain documents.

# Use DirectoryLoader to scan a folder and apply PyPDFLoader to each matching PDF.
loader = DirectoryLoader(
    path="books",
    glob="*.pdf",
    loader_cls=PyPDFLoader
)

# Load all matching PDF documents into memory.
# docs = loader.load()

# Lazy load all matching PDF documents one at a time.
docs = loader.lazy_load() # Gives a generator object for lazy loading

# Number of documents/pages loaded from the folder.
# print(len(docs)) # no of pages totally
print(docs) # Generator object

print(len(list(docs)))

for doc in docs:
    # If it was load this would print all pages content at once but takes more time and memory
    print(doc.page_content) 
    print(doc.metadata)
    break # Stop after the first document to demonstrate lazy loading

# Show the content and metadata of the first loaded PDF page/document.
# print(docs[0].page_content)
# print(docs[0].metadata)
