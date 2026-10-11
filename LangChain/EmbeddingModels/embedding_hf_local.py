from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

text = "Paris is the capital of France"
result = embeddings.embed_query(text)
print(str(result))

docs = [
    "Paris is the capital of France",
    "France is a country in Europe",
    "Europe is a continent",
    "The Earth is a planet in the Solar System"
]

results = embeddings.embed_documents(docs)
print(str(results))