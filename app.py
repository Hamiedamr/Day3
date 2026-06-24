import asyncio
import os
import streamlit as st

from ingest import ingest
from agent import create_rag_agent, stream_agent_response
from qdrant_client import QdrantClient

st.title("Agentic RAG over MCP (HTTP)")

# Path to Sohyla's CV (pre-loaded in the documents folder)
DEFAULT_CV_PATH = os.path.join("documents", "Sohyla_Gomaa_CV.pdf")
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "documents"

def run_async(coro):
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(coro)

def is_collection_populated():
    """Check if Qdrant collection already has data."""
    try:
        client = QdrantClient(url=QDRANT_URL)
        if client.collection_exists(COLLECTION_NAME):
            info = client.get_collection(COLLECTION_NAME)
            return info.points_count > 0
    except Exception:
        pass
    return False

if "agent" not in st.session_state:
    st.session_state.agent = None

# Auto-ingest Sohyla's CV only if collection is empty
if "cv_ingested" not in st.session_state:
    if is_collection_populated():
        st.session_state.cv_ingested = True
        st.sidebar.success("✅ CV already loaded in Qdrant.")
    elif os.path.exists(DEFAULT_CV_PATH):
        with st.spinner("Ingesting Sohyla's CV into Qdrant..."):
            ingest(DEFAULT_CV_PATH)
        st.session_state.cv_ingested = True
        st.success("✅ Sohyla's CV has been ingested successfully!")
    else:
        st.warning(f"CV not found at {DEFAULT_CV_PATH}. Please upload it below.")
        st.session_state.cv_ingested = False

# Also allow uploading additional documents
uploaded_file = st.sidebar.file_uploader("Upload additional PDF document", type=["pdf"])

if uploaded_file:
    os.makedirs("documents", exist_ok=True)
    temp_path = os.path.join("documents", uploaded_file.name)
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    with st.spinner("Ingesting into Qdrant..."):
        ingest(temp_path)

    st.success("Document ingested! Make sure mcp_server.py is running.")

if st.session_state.agent is None:
    with st.spinner("Connecting to MCP server over HTTP..."):
        st.session_state.agent = run_async(create_rag_agent())

user_input = st.chat_input("Ask me anything about the document...")

if user_input:
    st.chat_message("user").write(user_input)

    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        
        async def stream_wrapper():
            full_text = ""
            async for token in stream_agent_response(
                st.session_state.agent,
                user_input,
                thread_id="session_001"
            ):
                full_text += token
                response_placeholder.markdown(full_text + "▌")
            response_placeholder.markdown(full_text)

        run_async(stream_wrapper())