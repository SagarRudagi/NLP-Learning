"""RunnableLambda topic: This runnable wraps a Python function inside a LangChain step so custom logic can be applied to data.
It is helpful when you want to transform, validate, or analyze model output before continuing the chain.

Program description: This script generates a joke, then uses a lambda-style function to count the number of words in the joke.
The final output contains both the joke and its word count.
"""

from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain.schema.runnable import RunnableSequence, RunnableParallel, RunnablePassthrough, RunnableLambda
from dotenv import load_dotenv

# Load environment variables such as API keys from a .env file.
load_dotenv()

# Custom Python function used inside a runnable.
def word_count(text):
    return len(text.split())

# Prompt to generate a joke on the given topic.
prompt = PromptTemplate(
    template="Write a joke on {topic}",
    input_variables=["topic"]
)

# Chat model used to generate the joke.
model = ChatOpenAI()

# Parse model output into a plain string.
parser = StrOutputParser()

# First stage: generate the joke.
seqChain = RunnableSequence(prompt, model, parser) # or prompt | model | parser

# Second stage: keep the joke and compute its word count in parallel.
parallelChain = RunnableParallel({
    "joke": RunnablePassthrough(),
    "word_count": RunnableLambda(word_count) # or RunnableLambda(lambda text: len(text.split()))
})

# Combine both stages into one final chain.
finalChain = RunnableSequence(seqChain, parallelChain) # or seqChain | parallelChain

# Invoke the chain and print both the joke and the computed word count.
print(finalChain.invoke({"topic": "programming"})) # Output is a dictionary so you can extract values using keys like "joke" and "word_count"
