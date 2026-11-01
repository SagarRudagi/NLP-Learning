from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
import requests
from dotenv import load_dotenv

load_dotenv()

# Tool Creation

@tool
def multiply(a:int, b:int) -> int:
    """Multiply two numbers"""
    return a * b

# print(multiply.invoke({"a": 3, "b": 4}))
# print(multiply.name,"\n",multiply.description,"\n",multiply.args)

# Tool Binding

llm = ChatOpenAI()

llm_with_tools = llm.bind_tools([multiply])


query = 'Can you multiply 6 and 10'
messages = [HumanMessage(content=query)]
response = llm_with_tools.invoke(messages) # This does not get you the direct result, but rather the tool call information
messages.append(response)
# print(response)
# print(response.tool_calls[0])


# Tool Execution

# print(multiply.invoke({"a": 6, "b": 10})) # This directly gives the result of the multiplication
tool_result = multiply.invoke(response.tool_calls[0])
# print(tool_result) # This wraps the tool call information into a direct invocation. This is a Tool Message

messages.append(tool_result)
# print(messages) # This shows the full conversation including the human message, the tool call response, and the tool result

result = llm_with_tools.invoke(messages).content
print(result)




