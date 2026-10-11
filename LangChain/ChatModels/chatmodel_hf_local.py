"""
Brief explanation:
This script loads a Hugging Face pipeline locally and wraps it in a LangChain chat model
so a prompt can be run without requiring a remote API call.

Author's note:
This example is useful for local experimentation with open-source LLMs in a development
environment that already has the required libraries installed.
"""

from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline

# Create a local Hugging Face pipeline for text generation, then wrap it in a chat model.
llm = HuggingFacePipeline.from_model_id(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation",
    pipeline_kwargs={"temperature": 0.7, "max_new_tokens": 256}
)

model = ChatHuggingFace(llm=llm)

# Ask the local model a simple question.
result = model.invoke("What is the Capital of France?")

print(result.content)
