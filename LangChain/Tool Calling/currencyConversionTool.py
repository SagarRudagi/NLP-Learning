# tool create
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.tools import InjectedToolArg
from typing import Annotated
from langchain_core.tools import tool
from dotenv import load_dotenv
import requests
import json

load_dotenv()


@tool
def get_conversion_factor(base_currency: str, target_currency: str) -> float:
  """
  This function fetches the currency conversion factor between a given base currency and a target currency
  """
  url = f'https://v6.exchangerate-api.com/v6/c754eab14ffab33112e380ca/pair/{base_currency}/{target_currency}'

  response = requests.get(url)

  return response.json()

@tool
def convert(base_currency_value: int, conversion_rate: Annotated[float, InjectedToolArg]) -> float:
  """
  given a currency conversion rate this function calculates the target currency value from a given base currency value
  """

  return base_currency_value * conversion_rate


llm = ChatOpenAI()

llm_with_tools = llm.bind_tools([get_conversion_factor, convert])

messages = [HumanMessage('What is the conversion factor between INR and USD, and based on that can you convert 10 inr to usd')]

response = llm_with_tools.invoke(messages)

messages.append(response)


for tool_call in response.tool_calls:
    # execute the 1st tool and get the value of conversion rate
    if tool_call['name'] == "get_conversion_factor":
        tool_result = get_conversion_factor.invoke(tool_call) # get the conversion factor result
        conversion_rate = json.loads(tool_result.content)['conversion_rate'] # extract the conversion rate from the tool result
        messages.append(tool_result)

    # execute the 2nd tool using the conversion rate from tool 1
    if tool_call['name'] == "convert":
        tool_call['args']['conversion_rate'] = conversion_rate # inject the conversion rate from tool 1
        tool_result = convert.invoke(tool_call) # execute the convert tool
        messages.append(tool_result)


print(llm_with_tools.invoke(messages).content)


# Agent on the top

from langchain.agents import initialize_agent, AgentType

currency_agent = initialize_agent(
   tools=[get_conversion_factor, convert],
   llm=llm,
   agent_type=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION, # Using ReAct with structured chat
   verbose=True # Enable verbose mode to see the agent's reasoning steps
)

# Run the Agent
user_query = "Hi how are you?"

response = currency_agent.invoke({"input": user_query})