"""
Brief explanation:
This script demonstrates how to pass a list of system, user, and assistant messages to
LangChain's chat model to generate a response based on conversation context.

Author's note:
The example is intentionally minimal and serves as a quick introduction to working with
`SystemMessage`, `HumanMessage`, and `AIMessage` objects.
"""

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

# Load environment variables before creating the model.
load_dotenv()

# Create an OpenAI-backed chat model.
model = ChatOpenAI()

# Define a short conversation with both system instructions and user input.
messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="Tell me about LangChain python library"),
]

# Ask the model to respond using this conversation history.
response = model.invoke(messages)
messages.append(AIMessage(content=response.content))
print(messages)
