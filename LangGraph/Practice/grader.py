from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

class Grade(BaseModel):
    binary_score: str = Field(description="'yes' if relevant, 'no' if not")
    reason: str = Field(description="One sentence justifying the grade")

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
grader = llm.with_structured_output(Grade)

result = grader.invoke(
    "Question: How should employees behave when dealing with confidential customer information?\n"
    "Document: Employees must maintain professional conduct at all times.\n"
    "Is the document relevant to the question?"
)
print(result.binary_score, "-", result.reason)  # 'no' - ...