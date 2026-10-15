"""Demonstrates a simple LangChain pipeline with prompt, model, and parser.

This learning example creates a single chain that generates dialogues for a
character, parses the model response as plain text, and prints the chain graph.
"""

from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load environment variables such as OPENAI_API_KEY from .env.
load_dotenv()

# Prompt template with one dynamic input variable: character.
prompt = PromptTemplate(
    template='Generate 5 epic dialogues of {character}',
    input_variables=['character']
)

# Chat model used to generate the response.
model = ChatOpenAI()

# Parses AIMessage output into a plain Python string.
parser = StrOutputParser()

# Chain flow: prompt -> model -> string parser.
chain = prompt | model | parser

# Invoke the chain with input data.
result = chain.invoke({'character': 'Raymond Reddington'})

print(result)

# Visualize the chain structure
chain.get_graph().print_ascii()