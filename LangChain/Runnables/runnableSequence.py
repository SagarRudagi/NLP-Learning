"""RunnableSequence topic: This type of runnable executes one step after another in a strict sequence. It is useful when the output of one stage becomes the input for the next stage. 
IMPORTANT: This was being used everywhere, so the LangChain team made it simpler to use it by LCEL where now you can just use the pipe (|) operator to chain runnables together.

Program description: This script creates a two-step sequence where a prompt generates a joke,
then a second prompt explains that joke using the generated output as input.
"""

from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain.schema.runnable import RunnableSequence
from dotenv import load_dotenv

# Load environment variables such as API keys from a .env file.
load_dotenv()

# First prompt: generate the initial joke based on a topic.
prompt = PromptTemplate(
    template="Write a joke on {topic}",
    input_variables=["topic"]
)

# Second prompt: explain the joke that was just generated.
prompt2 = PromptTemplate(
    template="Explain this joke - {joke}",
    input_variables=["joke"]
)

# Chat model used for both generation steps.
model = ChatOpenAI()

# Parse each model output into a plain string before passing it to the next step.
parser = StrOutputParser()

# Build a sequential chain: prompt -> model -> parser -> prompt2 -> model -> parser.
chain = RunnableSequence(prompt, model, parser, prompt2, model, parser) # or using the pipe operator: chain = prompt | model | parser | prompt2 | model | parser

# Invoke the full sequence with a topic.
print(chain.invoke({"topic": "programming"}))
