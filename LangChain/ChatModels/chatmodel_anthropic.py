from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv

load_dotenv()

model = ChatAnthropic(model="claude-sonnet-4-6")

result = model.invoke("What is the Capital of France?")

print(result.content)