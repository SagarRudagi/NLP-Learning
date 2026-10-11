"""
Brief explanation:
This example shows how to include prior conversation history in a LangChain prompt
using `MessagesPlaceholder`, which is useful for chatbots that need context from earlier turns.

Author's note:
This script is intended as a simple reference for building multi-turn prompts in
LangChain while keeping the system message, history, and current user query together.
"""

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Create a chat template that adds a system message, then inserts stored conversation history,
# and finally appends the latest user question to the prompt.
chat_template = ChatPromptTemplate.from_messages(
    [
        ('system', "You are a helpful customer service agent."),
        MessagesPlaceholder(variable_name="chat_history"),
        ('human', '{query}')
    ]
)

# Load previous messages from a text file so the model has context from earlier conversations.
chat_history = []

with open('chat_history.txt') as f:
    chat_history.extend(f.readlines())

print(chat_history)

# Pass both the message history and the new query into the prompt.
prompt = chat_template.invoke({"chat_history": chat_history, "query": "What is the status of my refund?"})

print(prompt)
