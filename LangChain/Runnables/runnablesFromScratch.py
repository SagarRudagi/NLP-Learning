"""A minimal runnable chain built from scratch.

This program demonstrates how LangChain-style runnable components can be
assembled manually: a prompt template formats input, a dummy LLM generates a
response, and a string output parser extracts the final text.

This is a simpler custom version of the chain concept used in LangChain, and it
helps explain how the internal chain flow works in a more beginner-friendly way.
This file works to better the chainsFromScratch program by showing the same
core idea in a cleaner runnable pattern.
"""

from abc import ABC, abstractmethod
import random


class Runnable(ABC):
    # Abstract base class for all runnable components.
    # Any subclass must implement invoke() and override predict() or format() if needed to support the chain pipeline.
    @abstractmethod
    def invoke(self, input_data):
        pass


class DummyLLM(Runnable):
    def __init__(self):
        print('LLM Created')

    # Simulates an LLM response by returning a random sentence from a list.
    def invoke(self, prompt):
        response_list = [
            "The weather is sunny and perfect for a walk.",
            "A quick review shows the plan is on track and efficient.",
            "The new idea is creative, practical, and worth testing."
        ]
        return {'response': random.choice(response_list)}

    # Backward-compatible method kept as a warning for older usage.
    def predict(self, prompt):
        print("Warning: predict() is deprecated. Use invoke() instead.")

        response_list = [
            "The weather is sunny and perfect for a walk.",
            "A quick review shows the plan is on track and efficient.",
            "The new idea is creative, practical, and worth testing."
        ]
        return {'response': random.choice(response_list)}


class DummyPromptTemplate(Runnable):
    def __init__(self, template, input_variables):
        self.template = template
        self.input_variables = input_variables

    # Formats a template string using the input dictionary.
    def invoke(self, input_dict):
        return self.template.format(**input_dict)

    # Backward-compatible method kept for compatibility with older code.
    def format(self, input_dict):
        print("Warning: format() is deprecated. Use invoke() instead.")
        return self.template.format(**input_dict)


class DummyStrOutputParser(Runnable):
    def __init__(self):
        print('Output Parser Created')

    # Extracts the final response text from the LLM output dictionary.
    def invoke(self, input_data):
        return input_data['response']


class RunnableConnector(Runnable):
    def __init__(self, runnable_list):
        self.runnable_list = runnable_list

    # Runs each component in the list in sequence.
    def invoke(self, input_data):
        for runnable in self.runnable_list:
            input_data = runnable.invoke(input_data)
        return input_data

# USE CASE 1:
# Simple message generation using a prompt template, dummy LLM, and output parser.
# This shows the basic flow of data through a chain without using a real model.
# The flow works but the output is just random sentences as we provided.
print("Running Use Case 1: Simple message generation")

# Create a prompt template that formats a greeting message.
template = DummyPromptTemplate(
    template="Hello, {name}! Today is {day}.",
    input_variables=["name", "day"]
)

# Create a dummy LLM and output parser.
llm = DummyLLM()
parser = DummyStrOutputParser()

# Build a chain of runnable components.
chain = RunnableConnector([template, llm, parser])

# Send sample input through the chain and print the output.
input_data = {"name": "Alice", "day": "Monday"}
output = chain.invoke(input_data)
print(output)


# USE CASE 2:
# This is to demonstrate chaining two runnable pipelines together.
# The output of the first chain becomes the input for the second chain.
# Here, one chain generates a joke about a topic, and the second explains it.
# The flow works but the output is just random sentences as we provided.
print("Running Use Case 2: Connecting two chains: Joke generation and explanation")

# First prompt: generate a joke about a topic.
template1 = DummyPromptTemplate(
    template="Write a joke about {topic}",
    input_variables=["topic"]
)

# Second prompt: explain the generated joke.
template2 = DummyPromptTemplate(
    template="Explain the following joke {response}",
    input_variables=["response"]
)

# Reuse the same dummy LLM and parser for demonstration purposes.
llm = DummyLLM()
parser = DummyStrOutputParser()

# First chain creates the joke.
chain1 = RunnableConnector([template1, llm])

# Second chain explains the joke and extracts the final text.
chain2 = RunnableConnector([template2, llm, parser])

# Combine both chains so the output of chain1 feeds into chain2.
# This is the core idea of sequential chaining in LangChain.
final_chain = RunnableConnector([chain1, chain2])
output = final_chain.invoke({"topic": "chickens"})
print(output)