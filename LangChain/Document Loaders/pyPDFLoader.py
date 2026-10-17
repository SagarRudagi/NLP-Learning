"""PyPDFLoader example for loading a PDF into LangChain documents.
Limitations:
- This loader works best with text-based PDFs. Scanned images or non-text PDFs may not be parsed correctly.

Other PDF Loaders with their specific use cases:
Simple/ clean PDFs - PyPDFLoader
PDF with tables/columns - PDFPlumberLoader
Scanned/ image-based PDFs (with/without text) - UnstructuredPDFLoader/AmazonTextractPDFLoader
Need layout and image date - PyMuPDFLoader


This script shows how a PDF file can be parsed into Document objects, which can
then be used for later text processing, summarization, or question answering.
"""

from langchain_community.document_loaders import PyPDFLoader

# Load the PDF file into a document loader.
loader = PyPDFLoader("gcpServices.pdf")

# Extract the PDF content as a list of LangChain Document objects.
docs = loader.load()

# Number of loaded documents/pages returned by the loader.
print(len(docs))

# First document's content from the PDF.
print(docs[0].page_content)

# Metadata associated with the document, such as source file details.
print(docs[0].metadata)