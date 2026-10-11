import os
from pathlib import Path

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).with_name(".env"))

hf_token = (
    os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
)

if not hf_token:
    raise RuntimeError(
        "Missing Hugging Face token. Set HUGGINGFACEHUB_API_TOKEN (or HF_TOKEN) in .env."
    )

llm = HuggingFaceEndpoint(
    repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation",
    huggingfacehub_api_token=hf_token,
)

model = ChatHuggingFace(llm=llm)

result = model.invoke("What is the Capital of France?")

print(result.content)