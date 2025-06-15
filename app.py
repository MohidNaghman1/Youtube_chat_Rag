import streamlit as st
import re
import os
import requests
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain.text_splitter import RecursiveCharacterTextSplitter
# --- CHANGED: Import different Embeddings and LLM classes ---
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

# --- REMOVED: No longer need transformers, pipeline, or torch ---

st.set_page_config(page_title="YouTube RAG Chat", page_icon="🎥")

st.title("🎥 YouTube Video Chatbot")
st.markdown("Chat with any YouTube video using HuggingFace APIs!")



import os
from dotenv import load_dotenv

# --- CORRECTED SMART TOKEN LOADING ---

# Load environment variables from a .env file if it exists (for local development)
load_dotenv()

# Check if the app is running on Streamlit Community Cloud
# The 'STREAMLIT_SERVER_RUNNING_REMOTELY' env var is a reliable indicator
IS_RUNNING_ON_CLOUD = os.environ.get('STREAMLIT_SERVER_RUNNING_REMOTELY', '').lower() == 'true'

hf_token = None

if IS_RUNNING_ON_CLOUD:
    # We are on Streamlit Cloud, so we fetch the secret from st.secrets
    st.success("Running on Streamlit Cloud!", icon="☁️")
    try:
        hf_token = st.secrets["HUGGINGFACE_API_TOKEN"]
    except KeyError:
        st.error("HuggingFace API Token not found in Streamlit Cloud secrets!")
        st.stop()
else:
    # We are running locally, so we fetch the secret from the .env file
    st.success("Running locally!", icon="💻")
    hf_token = os.getenv('HUGGINGFACE_API_TOKEN')

# Final check to ensure the token was loaded
if not hf_token:
    st.error("HuggingFace API Token is not set. Please add it to your .env file locally or to your Streamlit Cloud secrets.")
    st.stop()

# --- (END OF CORRECTED SMART TOKEN LOADING) ---
# Initialize session state
if 'processed' not in st.session_state:
    st.session_state.processed = False
if 'main_chain' not in st.session_state:
    st.session_state.main_chain = None
if 'messages' not in st.session_state:
    st.session_state.messages = []

# --- Helper functions (Your original functions are mostly fine) ---

def extract_video_id(url):
    """Extract video ID from YouTube URL"""
    patterns = [
        r'(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/embed/)([^&\n?#]+)',
        r'youtube\.com/watch\?.*v=([^&\n?#]+)'
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None

def get_transcript(video_id):
    """Get transcript from YouTube video"""
    try:
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=['en'])
        transcript = ' '.join([d['text'] for d in transcript_list])
        return transcript
    except TranscriptsDisabled:
        st.error("Transcripts are disabled for this video.")
        return None
    except Exception as e:
        st.error(f"Error getting transcript: {str(e)}")
        return None

def format_docs(retrieve_docs):
    """Format documents for context"""
    context_text = "\n\n".join(doc.page_content for doc in retrieve_docs)
    return context_text

def query_huggingface(prompt, api_token):
    """Query HuggingFace Inference API directly"""
    API_URL = "https://api-inference.huggingface.co/models/HuggingFaceH4/zephyr-7b-beta"
    headers = {
        "Authorization": f"Bearer {api_token}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "inputs": prompt,
        "parameters": {
            "temperature": 0.1,
            "max_new_tokens": 512,
            "top_p": 0.95,
            "repetition_penalty": 1.1,
            "return_full_text": False
        }
    }
    
    try:
        response = requests.post(API_URL, headers=headers, json=payload)
        response.raise_for_status()  # Raise an exception for bad status codes
        
        if response.status_code == 503:
            # Model is loading, wait and retry
            st.warning("Model is loading, please wait a moment and try again...")
            return "The model is currently loading. Please try again in a few seconds."
            
        result = response.json()
        
        if isinstance(result, list) and len(result) > 0:
            return result[0].get("generated_text", "No response generated")
        elif isinstance(result, dict):
            return result.get("generated_text", "No response generated")
        else:
            return "Unexpected response format from the API"
            
    except requests.exceptions.RequestException as e:
        st.error(f"Error calling HuggingFace API: {str(e)}")
        return "Sorry, there was an error communicating with the AI model. Please try again."
    except (ValueError, KeyError) as e:
        st.error(f"Error processing API response: {str(e)}")
        return "Sorry, there was an error processing the AI model's response. Please try again."

@st.cache_resource
def create_rag_chain(transcript, api_token):
    """Process video transcript and create the RAG chain."""
    # Split text into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.create_documents([transcript])
    
    # Create embeddings using a local HuggingFace model
    st.info("Loading embedding model... (this may take a minute on first run)")
    embedding = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )
    st.info("Embedding model loaded.")
    
    # Create vector store
    vector_store = FAISS.from_documents(chunks, embedding)
    
    # Create retriever
    retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k': 4})
    
    # Create prompt template
    prompt_template = ChatPromptTemplate.from_template(
        """<s>[INST] You are an expert assistant for answering questions about YouTube videos.
    Use the following retrieved transcript context to answer the question.
    If you don't know the answer, just say that you don't know.
    Keep the answer concise and based on the provided context.

    CONTEXT:
    {context}

    QUESTION:
    {question} [/INST]</s>"""
    )
    
    # Create main chain
    def generate_response(input_data):
        context = format_docs(retriever.get_relevant_documents(input_data))
        prompt = prompt_template.format(context=context, question=input_data)
        return query_huggingface(prompt, api_token)
    
    return generate_response

# --- UI and Logic ---

# --- MODIFIED: Added logic to auto-fill URL from Chrome Extension ---
query_params = st.query_params
url_from_extension = query_params.get("youtube_url", [None])[0]
url = st.text_input("YouTube URL:", value=url_from_extension, placeholder="https://www.youtube.com/watch?v=...")
# --- (End of Modification) ---


if st.button("🔄 Process Video", type="primary"):
    if url:
        video_id = extract_video_id(url)
        if video_id:
            with st.spinner("Getting transcript..."):
                transcript = get_transcript(video_id)
            if transcript:
                with st.spinner("Building RAG chain... (this happens once per video)"):
                    st.session_state.main_chain = create_rag_chain(transcript, hf_token)
                    st.session_state.processed = True
                    st.session_state.messages = [] # Clear previous chat
                    st.success("✅ Video processed! Ask questions below.")
                    st.video(url)
                    st.info(f"Transcript length: {len(transcript)} characters")
        else:
            st.error("❌ Invalid YouTube URL")
    else:
        st.error("❌ Please enter a YouTube URL")

# Chat interface
if st.session_state.processed:
    st.markdown("---")
    st.subheader("💬 Chat with the Video")
    
    # Display messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask about the video..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = st.session_state.main_chain(prompt)
                    st.write(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    st.error(f"Error generating response: {str(e)}")

# Instructions are always good
if not st.session_state.processed:
    st.markdown("---")
    st.info("Paste a YouTube URL above and click 'Process Video' to start.")