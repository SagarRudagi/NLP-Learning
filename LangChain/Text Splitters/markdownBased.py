"""Example of using a Markdown-aware text splitter.

This script demonstrates how a Markdown document can be split into meaningful
chunks while preserving headings, lists, and code blocks. The splitter is
language-aware, so it respects Markdown structures rather than breaking content
arbitrarily at character limits.
"""

from langchain.text_splitter import RecursiveCharacterTextSplitter, Language

# Markdown content that we want to split into manageable chunks for LLM processing.
text = """
# Project Name: Smart Student Tracker

A simple Python-based project to manage and track student data, including their grades, age, and academic status.


## Features

- Add new students with relevant info
- View student details
- Check if a student is passing
- Easily extendable class-based design


## 🛠 Tech Stack

- Python 3.10+
- No external dependencies


## Getting Started

1. Clone the repo  
   ```bash
   git clone https://github.com/your-username/student-tracker.git

"""

# RecursiveCharacterTextSplitter.from_language is used with Language.MARKDOWN so the
# splitter understands Markdown boundaries such as headings, lists, and fenced code blocks.
markdown_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.MARKDOWN,
    chunk_size=200,
    chunk_overlap=0
)

# Split the Markdown text into chunks that maintain document structure.
markdown_chunks = markdown_splitter.split_text(text)
print(len(markdown_chunks))
print(markdown_chunks[0])
