from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import streamlit as st
from langchain_core.prompts import PromptTemplate, load_prompt

load_dotenv()
model = ChatOpenAI()

st.header('Reasearch Tool')

paper_input = st.selectbox( "Select Research Paper Name", ["Attention Is All You Need", "BERT: Pre-training of Deep Bidirectional Transformers", "GPT-3: Language Models are Few-Shot Learners", "Diffusion Models Beat GANs on Image Synthesis"] )

style_input = st.selectbox( "Select Explanation Style", ["Beginner-Friendly", "Technical", "Code-Oriented", "Mathematical"] ) 

length_input = st.selectbox( "Select Explanation Length", ["Short (1-2 paragraphs)", "Medium (3-5 paragraphs)", "Long (detailed explanation)"] )

# Template

template = load_prompt('template.json')


#Without Using Chains

# prompt = template.invoke({
#     'paper_input': paper_input,
#     'style_input': style_input,
#     'length_input': length_input
# })


# if st.button("Summarize"):
#     response = model.invoke(prompt)
#     st.write(response.contents)


# Using chains - You don't need to call invoke twice

if st.button("Summarize"):
    chain = template | model
    response = chain.invoke({
        'paper_input': paper_input,
        'style_input': style_input,
        'length_input': length_input
    })
    st.write(response.contents)