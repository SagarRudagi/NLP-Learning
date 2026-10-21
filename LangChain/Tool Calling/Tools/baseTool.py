from langchain.tools import BaseTool
from typing import Type
from pydantic import BaseModel, Field

# arg schema using pydantic

class MultiplyInput(BaseModel):
    a: int = Field(description="The first number to add") # Since there is no default value, Pydantic treats this field as required
    b: int = Field(description="The second number to add") # Since there is no default value, Pydantic treats this field as required


class MultiplyTool(BaseTool):
    name: str = "multiply"
    description: str = "Multiply two numbers"
    args_schema: Type[BaseModel] = MultiplyInput

    def _run(self, a: int, b: int) -> int:
        return a * b

multiply_tool = MultiplyTool()
response = multiply_tool.invoke({"a": 3, "b": 4})
print(response)