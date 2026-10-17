"""RunnableBranch topic: This runnable chooses one path based on a condition, similar to an if/else statement.
It is useful when you want different follow-up actions depending on the value or size of the current output.

Program description: This script creates a report, checks whether it is long enough, and either summarizes it or passes it along unchanged.
The branch behavior adds conditional logic to the chain flow.
"""

from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain.schema.runnable import RunnableSequence, RunnablePassthrough, RunnableBranch
from dotenv import load_dotenv

# Load environment variables such as API keys from a .env file.
load_dotenv()

# Prompt to generate the initial report.
prompt = PromptTemplate(
    template="Write a report on {topic}",
    input_variables=["topic"]
)

# Prompt to summarize the report only when the branch condition is true.
prompt2 = PromptTemplate(
    template="Summarize this report - {report}",
    input_variables=["report"]
)

# Chat model used for both generation stages.
model = ChatOpenAI()

# Parse model output into plain text.
parser = StrOutputParser()

# First stage: generate the report.
seqChain = RunnableSequence(prompt, model, parser) # or prompt | model | parser

# Branch logic:
# If the report has more than 300 words, summarize it.
# Otherwise, pass the report through unchanged.
branchChain = RunnableBranch(
    (lambda x: len(x.split()) > 300, RunnableSequence(prompt2, model, parser)), # or prompt2 | model | parser
    RunnablePassthrough()
)

# Combine report generation and conditional branching.
finalChain = RunnableSequence(seqChain, branchChain) # or seqChain | branchChain

# Invoke the final chain with a topic.
print(finalChain.invoke({"topic": "Artificial Intelligence"}))