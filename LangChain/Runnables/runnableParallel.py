"""RunnableParallel topic: This type of runnable runs multiple branches at the same time and combines their outputs.
It is useful when you want to generate different results from the same input without waiting for one to finish before the next begins.

Program description: This script creates two parallel branches that generate a tweet and a LinkedIn post for the same topic.
The result is returned as a dictionary with separate keys for each branch.
"""

from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain.schema.runnable import RunnableSequence, RunnableParallel
from dotenv import load_dotenv

# Load environment variables such as API keys from a .env file.
load_dotenv()

# First prompt: generate a short tweet on the topic.
prompt = PromptTemplate(
    template="Generate a tweet on {topic}",
    input_variables=["topic"]
)

# Second prompt: generate a LinkedIn post on the same topic.
prompt2 = PromptTemplate(
    template="Generate a linkedin post on {topic}",
    input_variables=["topic"]
)

# Chat model used in both branches.
model = ChatOpenAI()

# Parse both model outputs as plain strings.
parser = StrOutputParser()

# Run both prompts in parallel and return a dictionary of outputs.
parallel_chain = RunnableParallel({
    "tweet": RunnableSequence(prompt, model, parser), # or prompt | model | parser
    "linkedin": RunnableSequence(prompt2, model, parser) # or prompt2 | model | parser
})

# Invoke with one topic and print the combined result.
print(parallel_chain.invoke({"topic": "programming"})) # Output is a dictionary so you can extract values using keys like "tweet" and "linkedin"
