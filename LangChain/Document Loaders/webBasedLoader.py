"""
WebBasedLoader is used for loading and parsing web-based documents.
Uses BeautifulSoup for parsing web-based documents.
Limitations:
- Only supports web-based documents.
- May not handle dynamic content loaded via JavaScript.
- Network latency can affect loading times.
- Requires internet access to fetch documents.
- May not work with websites that have strict anti-scraping measures.
"""

from unittest import result

from langchain_community.document_loaders import WebBaseLoader
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
    template='Answer this question: {question} using this - \n {doc}',
    input_variables=['question', 'doc']
)

url = "https://reference.langchain.com/python/langchain-openai/chat_models/base/ChatOpenAI"

loader = WebBaseLoader(url)

# Load the web-based document into memory
docs = loader.load()

# Length of documents loaded from the web page
# print(len(docs)) # O/P: 1 because it is just one url

# print(docs[0].page_content) # Print the content of the first loaded web document

chain = prompt | model | parser

print(chain.invoke({'question': 'What is ChatOpenAI?', 'doc': docs[0].page_content}))