"""Demonstrates the usage of StrOutputParser with prompt templates in LangChain.

This example shows how to generate a detailed report, pass that output into a
second prompt to create a summary, and parse the model responses as plain
strings using StrOutputParser.
"""

from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load environment variables such as API keys from a .env file.
load_dotenv()

# Create the chat model to use for generation.
model = ChatOpenAI()

# First prompt: generate a detailed report on a given topic.
template1 = PromptTemplate(
    template='Write a detailed report on {topic}',
    input_variables=['topic']
)

# Second prompt: produce a 5-line summary from the generated report text.
# Note: '\n' is the correct newline escape sequence, not '/n'.
template2 = PromptTemplate(
    template='Write a 5 line summary on the following text.\n{text}',
    input_variables=['text']
)

# Parse model output as a plain string instead of a raw message object.
parser = StrOutputParser()

# Chain: report -> parse -> summary prompt -> model -> parse result.
# StrOutputParser makes it possible to pass output from one stage directly into the next which is not possible in traditional response.content handling.
chain = template1 | model | parser | template2 | model | parser

# Invoke the pipeline with a topic and print the final summary.
response = chain.invoke({'topic': 'Artificial Intelligence'})
print(response)