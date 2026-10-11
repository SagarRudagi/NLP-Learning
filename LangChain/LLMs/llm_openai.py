"""
Brief explanation:
This example uses the legacy OpenAI LLM wrapper to send a simple prompt and print the
completion returned by the model.

Author's note:
This script is a minimal introduction to calling a LangChain LLM directly without a
chat-based message structure.
"""

from langchain_openai import OpenAI
from dotenv import load_dotenv

# Load the environment and API key needed to authenticate with OpenAI.
load_dotenv()

# Create a text-generation model using the instruct-tuned GPT-3.5 API.
llm = OpenAI(model="gpt-3.5-turbo-instruct")

# Invoke the model with a straightforward question.
result = llm.invoke("What is the capital of France?")
print(result)
