"""
Brief explanation:
This script uses Anthropic's Claude chat model through LangChain so a simple prompt can
be sent to the model and its answer printed.

Author's note:
The example is intentionally simple and helps show how different providers use the same
LangChain chat interface.
"""

from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv

# Load environment variables before model creation.
load_dotenv()

# Create an Anthropic Claude chat model.
model = ChatAnthropic(model="claude-sonnet-4-6")

# Send a short prompt and print the response content.
result = model.invoke("What is the Capital of France?")

print(result.content)
