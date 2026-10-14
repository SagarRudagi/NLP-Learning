"""Example of structured output parsing with a TypedDict schema."""

# Author: This is a simple example of using TypedDict for structured output parsing. This can be only used to structure the output but cannot be validated as seen in the program. For validation purpose, use Pydantic. 

from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from typing import TypedDict, Annotated, Optional, Literal

# Load environment variables before creating the model client.
load_dotenv()

# Initialize the chat model used for structured output.
model = ChatOpenAI()


product_review = """
The wireless noise-cancelling headphone delivers a strong all-around experience with a balanced sound profile, effective active noise cancellation, and a comfortable fit that makes it easy to wear for long listening sessions. In everyday use, it performs reliably for music, calls, commuting, and focused work, offering a premium feel without being overly complicated to use.

Pros:
- Excellent noise cancellation that noticeably reduces background distractions
- Clear, detailed audio with a pleasant balance across bass, mids, and highs
- Comfortable earcups and headband for extended wear
- Long battery life that supports travel and full workdays
- Intuitive controls and stable wireless connectivity

Cons:
- Premium pricing may be a barrier for budget-conscious buyers
- Bass lovers may wish for a stronger low-end punch
- Bulkier than some compact travel headphones
- Some advanced features may require the companion app to unlock fully

Overall, this is a highly capable headphone that excels in comfort, sound quality, and noise reduction. It is a particularly good choice for frequent travelers, remote workers, and anyone who wants a dependable pair of headphones for daily use.
"""


class Review(TypedDict):
    # Basic example of a typed dict to structure a model's output
    # summary: str
    # sentiment: str

    # If there is an ambiguity regarding what the variable should mean, you can provide additional context or annotations to clarify it. Here example, if you can explain what the summary represents, you can add an annotation or comment to make it clear.

    key_themes: Annotated[list[str], "The main themes or points highlighted in the review"]
    summary: Annotated[str, "A brief summary of the review"]

    # You can make use of Literal to restrict the possible values to a predefined set
    sentiment: Annotated[Literal["pos", "neg"], "The overall sentiment of the review, e.g., positive, negative, neutral"]

    # Use Optional when a field may not always be present in the data
    pros: Optional[Annotated[list[str], "Provide here the positive aspects mentioned in the review"]]
    cons: Optional[Annotated[list[str], "Provide here the negative aspects mentioned in the review"]]

# Bind the TypedDict schema so the model returns structured data.
structured_model = model.with_structured_output(Review)

# Ask the model to analyze a review.
response = structured_model.invoke(product_review)

print(response)

