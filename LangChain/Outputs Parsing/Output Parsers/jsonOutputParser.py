"""Demonstrates JSON output parsing using a OpenAI chat model.

This example shows how to ask an OpenAI chat model for structured JSON output by providing
format instructions to the prompt and then parsing the response with
JsonOutputParser. This parser does not enforce a strict schema but ensures the output is valid JSON. The schema depends on the model used and the instructions provided.
"""

from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

# Load environment variables such as API keys from a .env file.
load_dotenv()

# Create the chat model to use for generation.
model = ChatOpenAI()

# JsonOutputParser instructs the model to return valid JSON and parses it.
parser = JsonOutputParser()

# The prompt includes instructions telling the model how to format its response.
# The parser's format instructions are injected into the prompt automatically.
template = PromptTemplate(
    template='Give me 5 facts about {topic} \n {format_instruction}',
    input_variables=['topic'], # if you have an empty input variable list, then while invoking you still have to provide an empty dictionary for input variables
    partial_variables={'format_instruction': parser.get_format_instructions()} # Injects JSON format instructions into the prompt automatically
    # get_format_instructions() provides the necessary instructions for the model to provide output for the type of parser used
)

# Chain: prompt -> model -> JSON parser.
chain = template | model | parser

# Invoke the chain with a topic and print the parsed JSON result.
result = chain.invoke({'topic': 'black hole'})

print(result) # Outputs a python dictionary parsed from the JSON response
