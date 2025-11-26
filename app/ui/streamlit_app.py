import streamlit as st
import sys
import os
import shutil

#importing project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from app.orchestrator import RetailInsightsOrchestrator

st.set_page_config(page_title="Insights Assistant", layout="wide")
st.title(" Retail Insights Assistant")

st.sidebar.header("Configuration")
mode = st.sidebar.radio("Mode", ["Summarization", "Conversational Q&A"])

#folder path for data
FOLDER_PATH = os.path.join(project_root, "data", "Sales Dataset")

VECTOR_DB_DIR = "vector_db" #folder to store vector

#button to delete and rebuild vector store
if st.sidebar.button("Rebuild Vector Store"):
    if os.path.exists(VECTOR_DB_DIR):
        shutil.rmtree(VECTOR_DB_DIR)
        st.success("Vector DB deleted. Restart the app to re-index.")
    else:
        st.info("No existing vector DB found – nothing to delete.")
@st.cache_resource
def get_orchestrator():
    return RetailInsightsOrchestrator(folder_path=FOLDER_PATH)


orchestrator = get_orchestrator()

question = None
if mode == "Conversational Q&A":
    question = st.text_input(
        "Ask a question (e.g., 'Which region had highest YoY growth in Q3?')",
        placeholder="Type your business question here...",
    )
    if st.button("Run Q&A"):
        if not question:
            st.warning("Please enter a question.")
        else:
            with st.spinner("Retrieving context and answering using RAG + agents..."):
                answer = orchestrator.conversational_qa(question)
            st.subheader("Answer")
            st.write(answer)

if mode == "Summarization":
    if st.button("Run summary"):
        with st.spinner("Generating summary using RAG + agents..."):
            summary = orchestrator.summarization_mode()
        st.subheader("Summary")
        st.write(summary)


   
