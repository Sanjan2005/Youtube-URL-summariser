import os
import validators,streamlit as st
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader,UnstructuredURLLoader
from langchain.prompts import PromptTemplate
st.set_page_config(page_title="YouTube Summarizer", page_icon=":video_camera:", layout="wide")
st.title("LangChain based YouTube/URL Summarizer :video_camera:")
st.subheader("Summarize YouTube videos or any URL content using LangChain and Groq")


with st.sidebar:
    groq_api_key = st.text_input("Groq API Key",value="",type="password")



prompt_templete = """You are a helpful assistant that summarizes content. 
    Please provide a concise summary of the following content and provide it in 350 words: {text}"""

prompt = PromptTemplate(input_variables=["text"], template=prompt_templete)

url = st.text_input("URL",label_visibility="collapsed")

if st.button("Summarize"):
    if not groq_api_key:
        st.error("Please enter your Groq API Key.")
    elif not url or not validators.url(url):
        st.error("Please enter a valid URL to start.")
    else:
        try:
            llm = ChatGroq(
                    model="llama-3.1-8b-instant",
                    api_key=groq_api_key
                    )
            with st.spinner("Loading"):
                if "youtube.com" in url:
                    loader=YoutubeLoader.from_youtube_url(url,add_video_info=True)
                else:
                    loader=UnstructuredURLLoader(urls=[url],ssl_verify=False,
                                                 headers={"User-Agent": "Mozilla/5.0 (Linux; Android 10; SM-A505FN) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.120 Mobile Safari/537.36"})
                data= loader.load()

                chain = load_summarize_chain(llm,chain_type = "stuff",prompt = prompt)
                summary = chain.run(data)
                st.success(summary)
                st.success("Summary generated successfully!")
        except Exception as e:
            st.exception(f"An error occurred: {e}")
