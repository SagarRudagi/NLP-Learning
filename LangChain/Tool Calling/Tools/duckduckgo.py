from langchain_community.tools import DuckDuckGoSearchRun

duckSearch = DuckDuckGoSearchRun()

results = duckSearch.invoke("What's the weather in Bangalore, India?")

print(results)
