"""
Brief explanation:
This script creates a simple interactive chatbot using LangChain's chat model and a
message history list so the model can respond with context across multiple turns.

Author's note:
The example is designed to demonstrate how to maintain conversation context while
letting a user chat with the assistant in a loop.
"""

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv

# Load environment variables, such as the OpenAI API key.
load_dotenv()

# Initialize the chat model.
model = ChatOpenAI()

# Keep track of the ongoing conversation, starting with a system instruction.
chat_history = [
    SystemMessage(content="You are a helpful assistant.")
]

# Continue prompting the user until they type "exit".
while True:
    user_input = input("You: ")
    chat_history.append(HumanMessage(content=user_input))

    if user_input.lower() == "exit":
        break

    # Send the full chat history to the model to maintain conversational context.
    response = model.invoke(chat_history)
    chat_history.append(AIMessage(content=response.content))
    print("Bot:", response.content)

print(chat_history)
