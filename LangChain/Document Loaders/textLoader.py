"""Document loading and summarization example using TextLoader.

This script loads a text file into LangChain document objects, inspects the
loaded structure, and then uses the document content as input to a prompt that
asks the model to produce a summary.
"""

from langchain_community.document_loaders import TextLoader
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

# Load environment variables such as API keys from a .env file.
load_dotenv()

# Chat model used to generate the final summary.
model = ChatOpenAI()

# Convert the model output into a plain string.
parser = StrOutputParser()

# Prompt to summarize the text loaded from the document.
prompt = PromptTemplate(
    template='Write a summary for the following poem - \n {poem}',
    input_variables=['poem']
)

# Load the text file as a LangChain document.
loader = TextLoader("sample_topic.txt")

docs = loader.load()

# A document loader typically returns a list of Document objects.
print(type(docs)) # O/P: <class 'list'>

print(len(docs))

print(type(docs[0])) # O/P: <class 'langchain_core.documents.base.Document'>

print(docs[0]) # O/P: page_content=<content of the first document>, metadata=<metadata of the first document>

print(docs[0].page_content) # O/P: <content of the first document>

print(docs[0].metadata) # O/P: <metadata of the first document>

# Create the chain: prompt -> model -> string parser.
chain = prompt | model | parser

# Use the extracted page content as the poem text for summarization.
print(chain.invoke({'poem': docs[0].page_content}))
