"""Example of using a language-aware code splitter for Python source.

This script shows how a Python source file can be split into smaller chunks while
preserving code structure and readability. The splitter is language-aware, so it
uses Python-specific boundaries such as class and function definitions instead of
cutting the code arbitrarily.
"""

from langchain.text_splitter import RecursiveCharacterTextSplitter, Language

# Sample Python code that we want to split into meaningful logical chunks.
text = """
class Student:
    def __init__(self, name, age, grade):
        self.name = name
        self.age = age
        self.grade = grade  # Grade is a float (like 8.5 or 9.2)

    def get_details(self):
        return self.name

    def is_passing(self):
        return self.grade >= 6.0


# Example usage
student1 = Student("Aarav", 20, 8.2)
print(student1.get_details())

if student1.is_passing():
    print("The student is passing.")
else:
    print("The student is not passing.")

"""

# RecursiveCharacterTextSplitter.from_language is specifically designed for source code.
# It understands language-specific separators and preserves code structure better than a
# generic character-based splitter.
code_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,
    chunk_size=200,
    chunk_overlap=0
)

# Split the Python code into chunks that are easier to process or pass to a model.
code_chunks = code_splitter.split_text(text)
print(len(code_chunks))
print(code_chunks[0])