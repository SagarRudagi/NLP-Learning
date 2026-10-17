"""RunnablePassthrough topic: This runnable simply passes the input through without changing it.
It is useful when you want to keep the original value available while sending another value into a different branch.

Program description: This script first creates a joke, then passes the original joke value forward while also generating an explanation.
The final result is a dictionary containing both the joke and its explanation.
"""

from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain.schema.runnable import RunnableSequence, RunnableParallel, RunnablePassthrough
from dotenv import load_dotenv

# Load environment variables such as API keys from a .env file.
load_dotenv()

# Prompt to generate the joke.
prompt = PromptTemplate(
    template="Write a joke on {topic}",
    input_variables=["topic"]
)

# Prompt to explain the generated joke.
prompt2 = PromptTemplate(
    template="Explain this joke - {joke}",
    input_variables=["joke"]
)

# Chat model used for both generation steps.
model = ChatOpenAI()

# Parse model output to plain text.
parser = StrOutputParser()

# First sequence: create the joke.
joke_chain = RunnableSequence(prompt, model, parser) # or prompt | model | parser

# Second structure: keep the original joke value and produce an explanation in parallel.
explain_parallel_chain = RunnableParallel({
    "joke": RunnablePassthrough(),
    "explain": RunnableSequence(prompt2, model, parser)
})

# Combine both stages: joke generation followed by passthrough + explanation.
final_chain = RunnableSequence(joke_chain, explain_parallel_chain)

# Invoke the chain and print the final dictionary.
print(final_chain.invoke({"topic": "programming"})) # Output is a dictionary so you can extract values using keys like "joke" and "explain"