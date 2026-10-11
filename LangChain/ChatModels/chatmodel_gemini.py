"""
Brief explanation:
This example uses Google's Generative AI model through LangChain to send a prompt and
print the model's reply.

Author's note:
This file demonstrates how to swap model providers while keeping the same LangChain
chat invocation pattern.
"""

from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

# Load environment variables required for the Google API.
load_dotenv()

# Create a Gemini chat model instance.
model = ChatGoogleGenerativeAI(model="gemini-3-flash-preview")

# Invoke the model with a basic question.
result = model.invoke("What is the Capital of France?")

print(result.content)
