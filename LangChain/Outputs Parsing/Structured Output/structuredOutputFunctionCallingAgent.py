"""Example of function/tool calling with an agent using structured output.

This script demonstrates how an agent can decide to call a tool with a
well-defined schema, rather than returning free-form text. The result is then
processed and returned as structured data.
"""

# Author's Note: This script is written by Copilot. Just wanted to check how functions can be used as tools for function calling for agents in LangChain.

from dotenv import load_dotenv
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

load_dotenv()


@tool
def analyze_review(review_text: str, product_name: str) -> dict:
    """Analyze a product review and return structured insights."""
    normalized_text = review_text.lower()

    key_themes = []
    if "camera" in normalized_text:
        key_themes.append("camera")
    if "battery" in normalized_text:
        key_themes.append("battery")
    if "performance" in normalized_text or "processor" in normalized_text:
        key_themes.append("performance")
    if "price" in normalized_text or "expensive" in normalized_text:
        key_themes.append("price")
    if "design" in normalized_text or "weight" in normalized_text or "size" in normalized_text:
        key_themes.append("design")

    sentiment = "pos" if "amazing" in normalized_text or "great" in normalized_text else "neg"

    summary = (
        f"The review of {product_name} highlights strong performance and camera quality, "
        "but also mentions concerns about pricing and usability."
    )

    return {
        "product_name": product_name,
        "summary": summary,
        "sentiment": sentiment,
        "key_themes": key_themes,
        "pros": [
            "Fast performance",
            "Strong camera quality",
            "Good battery life",
        ],
        "cons": [
            "High price",
            "Large size",
            "Some usability trade-offs",
        ],
    }


llm = ChatOpenAI(model="gpt-4o-mini")

agent = create_tool_calling_agent(llm, [analyze_review])
agent_executor = AgentExecutor(agent=agent, tools=[analyze_review])

review_text = """
The Samsung Galaxy S24 Ultra is a powerhouse. The camera is stunning, the battery lasts all day,
and the performance is excellent. However, it is expensive and a bit bulky for one-handed use.
"""

result = agent_executor.invoke({
    "input": (
        f"Analyze this product review for {review_text} and provide a structured result."
    )
})

print(result["output"])
