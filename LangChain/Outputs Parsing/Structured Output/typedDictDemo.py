"""Demonstrates using TypedDict to define and type-check a simple person record."""

# Author: This demonstrates a basic example to use TypedDict which allows defining and type-checking structured dictionaries in Python. But if different datatype data is entered, it will still not raise an error at runtime.

from typing import TypedDict

class Person(TypedDict):
    name: str
    age: int

new_person: Person = {
    "name": "John Doe",
    "age": 30
}

print(new_person)