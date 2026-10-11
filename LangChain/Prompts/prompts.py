"""
Brief explanation:
This Streamlit app allows a user to choose a research paper, explanation style, and
length, then summarizes the paper with a LangChain prompt and model chain.

Author's note:
The script demonstrates how to combine a prompt template with a chat model in a simple
user-facing interface while keeping the code easy to follow.
"""

from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import streamlit as st
from langchain_core.prompts import PromptTemplate, load_prompt

# Load environment variables for the OpenAI API connection.
load_dotenv()
model = ChatOpenAI()

# Display the app header and let the user choose summary preferences.
st.header('Reasearch Tool')

paper_input = st.selectbox(
    "Select Research Paper Name",
    [
        "Attention Is All You Need",
        "BERT: Pre-training of Deep Bidirectional Transformers",
        "GPT-3: Language Models are Few-Shot Learners",
        "Diffusion Models Beat GANs on Image Synthesis"
    ]
)

style_input = st.selectbox(
    "Select Explanation Style",
    ["Beginner-Friendly", "Technical", "Code-Oriented", "Mathematical"]
)

length_input = st.selectbox(
    "Select Explanation Length",
    ["Short (1-2 paragraphs)", "Medium (3-5 paragraphs)", "Long (detailed explanation)"]
)

# Load the saved prompt template containing the generation instructions.
template = load_prompt('template.json')

# Without Using Chains:
# prompt = template.invoke({
#     'paper_input': paper_input,
#     'style_input': style_input,
#     'length_input': length_input
# })
# if st.button("Summarize"):
#     response = model.invoke(prompt)
#     st.write(response.contents)

# Using chains is the preferred pattern here because the prompt and model are composed in one pipeline.
if st.button("Summarize"):
    chain = template | model
    response = chain.invoke({
        'paper_input': paper_input,
        'style_input': style_input,
        'length_input': length_input
    })
    st.write(response.contents)
