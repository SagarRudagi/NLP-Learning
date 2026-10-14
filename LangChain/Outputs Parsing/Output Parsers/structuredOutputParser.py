"""Demonstrates structured output parsing with StructuredOutputParser.
NOTE: StructuredOutputParser does enforce a schema on the model output but it does not perform any data validation beyond the schema. This gap is filled by PydanticOutputParser.
This learning example defines a response schema, injects parser-generated
format instructions into the prompt, and converts the model output into a
structured Python dictionary.
"""

from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
# ATTENTION: StructuredOutputParser and ResponseSchema come from langchain.output_parsers, not langchain_core.output_parsers.
from langchain.output_parsers import StructuredOutputParser, ResponseSchema

# Load environment variables such as API keys from a .env file.
load_dotenv()

# Create the chat model to use for generation.
model = ChatOpenAI()

# Define the structure we want from the model output.
schema = [
    ResponseSchema(name='dialogue_1', description='The first dialogue of the character'),
    ResponseSchema(name='dialogue_2', description='The second dialogue of the character'),
    ResponseSchema(name='dialogue_3', description='The third dialogue of the character'),
]

# Build a parser that validates and parses output based on the schema above.
parser = StructuredOutputParser.from_response_schemas(schema)

# Inject parser format instructions so the model returns schema-compliant output.
template = PromptTemplate(
    template='Provide three famous dialogues of {character}: \n{format_instructions}',
    input_variables = ['character'],
    partial_variables={'format_instructions': parser.get_format_instructions()}
)

# Chain: prompt -> model -> structured parser.
chain = template | model | parser

# Invoke with a character name and print the parsed dictionary result.
response = chain.invoke({'character': 'Sherlock Holmes'})
print(response)