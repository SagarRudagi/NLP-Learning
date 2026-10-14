"""Demonstrates structured output parsing with PydanticOutputParser. 
NOTE: PydanticOutputParser enforces both the schema and validation rules defined in the Pydantic model.

This learning example defines a Pydantic model as the output schema, injects
format instructions into the prompt, and parses model output into a validated
Pydantic object.
"""

from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

# Load environment variables such as API keys from a .env file.
load_dotenv()

# Create the chat model used for text generation.
model = ChatOpenAI()

# Define the expected structured output with validation rules.
class Person(BaseModel):
    name: str = Field(description='The name of the person')
    age: int = Field(gt=18, description='The age of the person')
    city: str = Field(description='The city where the person lives')

# Build a parser that converts model output into a validated Person object.
parser = PydanticOutputParser(pydantic_object=Person)

# Inject parser format instructions so the model responds in the required shape.
template = PromptTemplate(
    template = 'Provide name, age and city of a person living in {country}\n{format_instructions}',
    input_variables = ['country'],
    partial_variables={'format_instructions': parser.get_format_instructions()}
)

# Chain flow: prompt -> model -> Pydantic parser.
chain = template | model | parser

# Invoke the chain and print the validated Person object.
response = chain.invoke({'country': 'India'})
print(response)
