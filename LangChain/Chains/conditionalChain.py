"""Demonstrates conditional routing in LangChain using sentiment classification.

This example first classifies feedback as positive or negative using a
Pydantic parser, then routes to a different response prompt based on the
predicted sentiment.
"""

from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel, RunnableBranch, RunnableLambda
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal

# Load environment variables such as API keys from .env.
load_dotenv()

# Base chat model used for classification and response generation.
model = ChatOpenAI()

# Converts AIMessage output to plain text for final response branches.
strParser = StrOutputParser()

# Structured schema for sentiment classification output.
class Feedback(BaseModel):
    sentiment: Literal["positive", "negative"] = Field(description="The sentiment of the feedback")

# Parses model output into a validated Feedback object.
pydParser = PydanticOutputParser(pydantic_object=Feedback)

# Prompt for classification step with parser-generated formatting instructions.
feedback_prompt = PromptTemplate(
    template='Classify the sentiment of the feedback into positive or negative\n {feedback} \n {format_instructions}',
    input_variables=['feedback'],
    partial_variables={'format_instructions': pydParser.get_format_instructions()}
)

# Stage 1: feedback text -> model -> validated Feedback object.
classification_chain = feedback_prompt | model | pydParser

# Prompt used when sentiment is positive.
positive_feedback_response_prompt = PromptTemplate(
    template='Write an appropriate response for the positive feedback: {feedback}',
    input_variables=['feedback']
)

# Prompt used when sentiment is negative.
negative_feedback_response_prompt = PromptTemplate(
    template='Write an appropriate response for the negative feedback: {feedback}',
    input_variables=['feedback']
)

# Stage 2: route execution based on sentiment value from the Feedback object.
# - First branch runs when sentiment is positive.
# - Second branch runs when sentiment is negative.
# - Final RunnableLambda is the default fallback branch.
branch_chain = RunnableBranch(
    (lambda x:x.sentiment=='positive', positive_feedback_response_prompt | model | strParser),
    (lambda x:x.sentiment=='negative', negative_feedback_response_prompt | model | strParser),
    RunnableLambda(lambda x: "Could not classify feedback") # default chain
)

# Full flow: classification step followed by conditional response generation.
chain = classification_chain | branch_chain

# Example invocation with a single feedback input.
response = chain.invoke({'feedback': 'The product is great!'})
print(response)

# Visualize the runnable graph for better understanding.
chain.get_graph().print_ascii()
