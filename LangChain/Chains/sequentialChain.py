"""Demonstrates a sequential LangChain pipeline with two prompt stages.

This learning example first generates a detailed report on a topic, then feeds
that output into a second prompt to produce a concise 5-point summary.
"""

from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load environment variables such as API keys from .env.
load_dotenv()

# First prompt: produce a detailed report for the given topic.
prompt1 = PromptTemplate(
    template='Generate a detailed report on {topic}',
    input_variables=['topic']
)

# Second prompt: convert the report text into a 5-point summary.
prompt2 = PromptTemplate(
    template='Generate a 5 pointer summary from the following text \n {text}',
    input_variables=['text']
)

# Chat model used in both stages of the sequence.
model = ChatOpenAI()

# Converts model message outputs into plain strings between stages.
parser = StrOutputParser()

# Sequential flow:
# report prompt -> model -> parse -> summary prompt -> model -> parse.
chain = prompt1 | model | parser | prompt2 | model | parser

# Invoke the chain with a topic input.
result = chain.invoke({'topic': 'Unemployment in India'})

print(result)

# Print an ASCII visualization of the chain graph.
chain.get_graph().print_ascii()