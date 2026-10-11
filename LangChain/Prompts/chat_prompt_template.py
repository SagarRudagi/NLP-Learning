"""
Brief explanation:
This file shows how to build a reusable chat prompt template with dynamic values for
`domain` and `topic`, making it easy to customize responses for different contexts.

Author's note:
This is a straightforward template example for learning how message-based prompts are
constructed in LangChain.
"""

from langchain_core.prompts import ChatPromptTemplate

# Define a reusable chat prompt where both the role and the message content can change.
chat_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful {domain} expert."),
        ("human", "Explain in simple terms about {topic}"),
    ]
)

# Fill in the template variables before invoking the prompt.
prompt = chat_template.invoke({"domain": "AI", "topic": "LangChain"})
print(prompt)
