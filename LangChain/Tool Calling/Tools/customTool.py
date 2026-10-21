from langchain_core.tools import tool
# Step 1 - create a function
# Step 2 - add type hints
# Step 3 - add tool decorator

@tool
def multiply(a: int, b:int) -> int:
    """Multiply two numbers"""
    return a*b

# Step 4 - invoke the tool
response = multiply.invoke({"a": 3, "b": 4})
print(response)

print(multiply.args_schema.model_json_schema()) # This goes to LLM