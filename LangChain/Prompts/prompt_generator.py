"""
Brief explanation:
This script defines a reusable `PromptTemplate` that instructs the model to write a
research-paper summary in a specific style and length, then saves it for reuse.

Author's note:
The prompt is designed to encourage accurate summaries, include math where needed, and
fallback to a clear "Insufficient information available" message when the paper does not provide details.
"""

from langchain_core.prompts import PromptTemplate

# Build a structured prompt that tells the model how to summarize a paper.
template = PromptTemplate(
    template="""
    Please summarize the research paper titled \"{paper_input}\" with the following specifications:
    Explanation Style: {style_input}
    Explanation Length: {length_input}
    1. Mathematical Details:
        - Include relevant mathematical equations if present in the paper.
        - Explain the mathematical concepts using simple, intuitive code snippets where applicable.
    2. Analogies:
        - Use relatable analogies to simplify complex ideas.
        If certain information is not available in the paper, respond with: \"Insufficient information available\" instead of guessing.
        Ensure the summary is clear, accurate, and aligned with the provided style and length.
""",
    input_variables=["paper_input", "style_input", "length_input"],
    validate_template=True
)

# Save the prompt so other scripts can load and reuse it without redefining the template.
template.save('template.json')
