"""CSVLoader example for reading structured data as LangChain documents.

This script demonstrates how a CSV file can be loaded into Document objects so
its rows can be processed and analyzed by downstream LangChain workflows.
"""

from langchain_community.document_loaders import CSVLoader

# Load the CSV file into a document loader.
loader = CSVLoader(file_path='Social_Network_Ads.csv')

# Read the CSV and convert each row into a LangChain Document.
docs = loader.load()

# Print how many rows/documents were loaded.
print(len(docs))

# Show the second document entry as an example of parsed row content.
print(docs[1])