from langchain_community.retrievers import WikipediaRetriever

retriever = WikipediaRetriever(top_k_results=2, lang="en")

query = "What are the conspiracies of Adolf Hitler?"
results = retriever.invoke(query)
print(results)
# Print retrieved content
for i, doc in enumerate(results):
    print(f"\n--- Result {i+1} ---")
    print(f"Content:\n{doc.page_content}...")