import validators
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import UnstructuredURLLoader
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from langchain.schema import Document  

## Streamlit APP
st.set_page_config(page_title="LangChain: Summarize Text From YT or Website", page_icon="🦜")
st.title("🦜 LangChain: Summarize Text From YT or Website")
st.subheader('Summarize URL')

## Get the Groq API Key and URL (YT or website) to be summarized
with st.sidebar:
    groq_api_key = st.text_input("Groq API Key", value="", type="password")

generic_url = st.text_input("URL", label_visibility="collapsed")

## Groq Model (Gemma 2)
llm = ChatGroq(model="gemma2-9b-it", groq_api_key=groq_api_key)

prompt_template = """
Provide a summary of the following content in 300 words:
Content:{text}
"""
prompt = PromptTemplate(template=prompt_template, input_variables=["text"])

if st.button("Summarize the Content from YT or Website"):
    ## Validate inputs
    if not groq_api_key.strip() or not generic_url.strip():
        st.error("Please provide the necessary information to get started.")
    elif not validators.url(generic_url):
        st.error("Please enter a valid URL. It can be a YouTube video URL or a website URL.")
    else:
        try:
            with st.spinner("Processing..."):
                docs = []  

                ## Process YouTube Video
                if "youtube.com" in generic_url or "youtu.be" in generic_url:
                    try:
                        video_id = generic_url.split("watch?v=")[-1].split("&")[0]  # Extract video ID
                        transcript = YouTubeTranscriptApi.get_transcript(video_id)  # Fetch transcript
                        transcript_text = " ".join([t["text"] for t in transcript])  # Convert to plain text
                        docs = [Document(page_content=transcript_text)]  # Store in LangChain Document format
                    except (TranscriptsDisabled, NoTranscriptFound):
                        st.error("No transcript found for this YouTube video.")
                        st.stop()
                    except Exception as yt_error:
                        st.error(f"Error fetching transcript: {yt_error}")
                        st.stop()

                ## Process Website Content
                else:
                    try:
                        loader = UnstructuredURLLoader(
                            urls=[generic_url],
                            ssl_verify=False,
                            headers={
                                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                                              "AppleWebKit/537.36 (KHTML, like Gecko) "
                                              "Chrome/120.0.0.0 Safari/537.36"
                            }
                        )
                        docs = loader.load()  # Loader is only used for websites
                    except Exception as web_error:
                        st.error(f"Error fetching webpage content: {web_error}")
                        st.stop()

                ## Ensure content exists before summarization
                if not docs:
                    st.error("No content found to summarize. Please check the URL.")
                    st.stop()

                ## Chain for summarization
                chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt)
                output_summary = chain.run(docs)

                st.success(output_summary)

        except Exception as e:
            st.exception(f"Unexpected error: {e}")

                    