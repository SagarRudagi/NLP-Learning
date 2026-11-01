from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub
import requests
from dotenv import load_dotenv

load_dotenv()

search_tool = DuckDuckGoSearchRun()

@tool
def get_weather_data(city: str) -> str:
  """
  This function fetches the current weather data for a given city
  """
  url = f'https://api.weatherstack.com/current?access_key=4d1d8ae207a8c845a52df8a67bf3623e&query={city}'

  response = requests.get(url)

  return response.json()


llm = ChatOpenAI()

# Pull the ReAct prompt from LangChain Hub
prompt = hub.pull("hwchase17/react")  # pulls the standard ReAct agent prompt from LangChain Hub

# Create the ReAct agent using the pulled prompt and LLM
agent = create_react_agent(
  llm=llm,
  tools=[search_tool, get_weather_data],
  prompt=prompt
)

# Create an agent executor to run the agent
agent_executor = AgentExecutor(
  agent=agent,
  tools=[search_tool, get_weather_data],
  verbose=True
)

response = agent_executor.invoke({"input": "Find how tall is Empire State Building and What is the weather in New York?"})

print(response)
