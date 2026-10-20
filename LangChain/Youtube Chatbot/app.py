from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
import streamlit as st
from dotenv import load_dotenv

import os
# Workaround for OpenMP duplicate runtime initialization seen with FAISS on some macOS setups.
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

load_dotenv()

ytt_api = YouTubeTranscriptApi()

llm  = ChatOpenAI(temperature=0.2)

parser = StrOutputParser()

def format_context(docs):
    return "\n".join([doc.page_content for doc in docs])

form = st.form("yt_url")
video_url = form.text_input("Enter YouTube video URL...")
# video_url = "https://www.youtube.com/watch?v=wjZofJX0v4M&list=PLZHQObOWTQDN52m7Y21ePrTbvXkPaWVSg&index=8&t=1s"

question = form.text_input("Enter your question...")
url_submit = form.form_submit_button("Submit")
placeholder = st.empty()

if url_submit and video_url and question:
    placeholder.write("Processing your request...")
    video_id = video_url.split("v=")[1].split("&list")[0]

    try:
        transcript = ytt_api.fetch(video_id)
        transcript = " ".join(t.text for t in transcript)
    except TranscriptsDisabled:
        print("Transcripts are disabled for this video.")
    except:
        print("Provide valid URL.")


    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.create_documents([transcript])

    vectorstore = FAISS.from_documents(chunks, OpenAIEmbeddings())

    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})

    parallel_chain = RunnableParallel({
                'context': retriever | RunnableLambda(format_context),
                'question': RunnablePassthrough()
            })
    
    prompt_template = PromptTemplate(
        template="""You are a helpful assistant. Answer the following question with the given context. If the context is insufficient, response with 'I don't know.'" 
        Context: {context}
        Question: {question}
        """,
        input_variables=["context", "question"]
    )
        
    model_chain = parallel_chain | prompt_template | llm | parser

    response = model_chain.invoke(question)

    placeholder.write(response)

else:
    placeholder.write("Please enter a valid YouTube URL and question.")









