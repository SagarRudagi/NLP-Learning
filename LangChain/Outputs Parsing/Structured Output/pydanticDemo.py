"""Description: Example of validating structured data with a Pydantic model."""

# Author: This is a simple example demonstrating how to use Pydantic to validate structured data.

from pydantic import BaseModel, EmailStr, Field
from typing import Optional


class Student(BaseModel):
    # You can also give a default value if required
    name: str
    
    # You can also define an optional parameter but have to set a default value for it
    age: Optional[int] = None
    
    # EmailStr: This is built-in type provided by Pydantic for email validation.
    # Pydantic will validate that the email is in a proper format
    email: EmailStr

    # Field with constraints: Here we are using Pydantic's Field to enforce that cgpa must be greater than 0 and less than or equal to 10.
    # It also has a "default" parameter to provide a default value if needed.
    # Description: This field represents the student's CGPA and must be a float between 0 and 10. Acts like TypedDict in terms of type enforcement. Send's the description to the LLM if used.
    cgpa: float = Field(gt=0, le=10, description="A decimal value representing the student's CGPA")

# Raw input data that will be validated by the model. 
# Type coercion: Here even if age is given as a string, Pydantic will attempt to coerce it to the correct type. This does not guarantee that all types will be correctly converted, so validation errors may still occur.
new_student = {"name": "Tony Stark", "age": 45, "email": "tony.stark@example.com", "cgpa": 9.5}

# Create a Student instance from the dictionary using keyword unpacking.
student = Student(**new_student)

# Display the validated model instance. This will throw error if the name is of different type than defined earlier
# This provides a Pydantic object. You can explicitly convert to dict and use for further processing.
# Example: obj.model_dump() -> dict, obj.model_dump_json() -> JSON string
print(student)
