"""
Brief explanation:
This script configures a chat-based OpenAI model and asks a simple factual question to
show how a LangChain chat model returns content and metadata.

Author's note:
The example also explains how `temperature` and `max_tokens` affect generation behavior,
which is useful when tuning model outputs.
"""

from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

# Load the environment variables before instantiating the model.
load_dotenv()

# The model is created with a specific OpenAI chat model. The commented arguments show how
# to control response randomness and generation length.
model = ChatOpenAI(model="gpt-4")  # , temperature=0, max_tokens=100

"""
temperature -> controls the randomness of the model's output. Lower values make the output more deterministic, while higher values increase creativity.

max_tokens -> number of tokens to generate in the response.
"""

# Ask a simple question and print only the model's final message content.
result = model.invoke("What is the Capital of France?")

# print(result)
# Output: content='The capital of France is Paris.' additional_kwargs={...}

print(result.content)
# Output: The capital of France is Paris.
