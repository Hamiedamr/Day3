"""Streamlit UI (MCP Client) for the MCP-based Agentic RAG system.

The Streamlit app is now a thin MCP client + chat UI. It does NOT do
retrieval itself — it asks the agent, which calls the MCP server over HTTP.

Usage:
    uv run streamlit run app_mcp.py
"""

import asyncio
import os
import streamlit as st

from ingest import ingest
from agent import create_rag_agent, stream_agent_response

# ---- Persistent event loop (fixes "Event loop is closed" on Streamlit re-runs) ----
try:
    _loop = asyncio.get_running_loop()
except RuntimeError:
    _loop = asyncio.new_event_loop()
    asyncio.set_event_loop(_loop)


def _run_async(coro):
    """Run an async coroutine using the persistent event loop."""
    return _loop.run_until_complete(coro)


# Page title
st.title("🤖 Agentic RAG over MCP (HTTP)")

# Initialize session state for the agent
if "agent" not in st.session_state:
    st.session_state.agent = None

# --- Sidebar: upload + ingest ---
uploaded_file = st.sidebar.file_uploader("Upload a PDF document")

if uploaded_file:
    os.makedirs("documents", exist_ok=True)
    temp_path = os.path.join("documents", uploaded_file.name)
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    with st.spinner("Ingesting into Qdrant..."):
        # Run ingestion (loads PDF into Qdrant via native client)
        ingest(temp_path)

    st.success("Document ingested! Make sure mcp_server.py is running.")

# --- Build the agent once (connects to the MCP server over HTTP) ---
if st.session_state.agent is None:
    with st.spinner("Connecting to MCP server over HTTP..."):
        # create_rag_agent is async — run it from Streamlit
        st.session_state.agent = _run_async(create_rag_agent())

# --- Chat interface ---
user_input = st.chat_input("Ask a question about your document")

if user_input:
    st.chat_message("user").write(user_input)

    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""

        async def run_stream():
            global full_response
            async for token in stream_agent_response(
                st.session_state.agent,
                user_input,
                thread_id="session_001"
            ):
                full_response += token
                response_placeholder.markdown(full_response + "▌")

        # Drive the async stream from Streamlit
        _run_async(run_stream())
        response_placeholder.markdown(full_response)
