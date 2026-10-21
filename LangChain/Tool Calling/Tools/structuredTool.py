from langchain.tools import StructuredTool
from pydantic import BaseModel, Field

class MultiplyInput(BaseModel):
    a: int = Field(required=True, description="The first number")
    b: int = Field(required=True, description="The second number")

def multiply_func(a: int,b: int) -> int:
    return a * b

multiply_tool = StructuredTool.from_function(
    func=multiply_func,
    name="multiply",
    description="multiply two numbers",
    args_schema=MultiplyInput
)

response = multiply_tool.invoke({"a": 3, "b": 4})
print(response)