"""
Brief explanation:
This script connects to a Hugging Face model endpoint and uses a LangChain chat wrapper
to ask a basic question through a hosted text-generation API.

Author's note:
This example is useful when you want to test a remote Hugging Face model without relying
on a locally downloaded model checkpoint.
"""

import os
from pathlib import Path

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

# Load environment variables from a local .env file next to this script.
load_dotenv(dotenv_path=Path(__file__).with_name(".env"))

# Read the Hugging Face token from the environment.
hf_token = os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")

if not hf_token:
    raise RuntimeError(
        "Missing Hugging Face token. Set HUGGINGFACEHUB_API_TOKEN (or HF_TOKEN) in .env."
    )

# Configure the remote text-generation model endpoint.
llm = HuggingFaceEndpoint(
    repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation",
    huggingfacehub_api_token=hf_token,
)

# Wrap the endpoint in a LangChain chat model so it behaves like a standard chat interface.
model = ChatHuggingFace(llm=llm)

# Ask the hosted model a simple question.
result = model.invoke("What is the Capital of France?")

print(result.content)
