"""Streamlit UI for the Agentic RAG system over MCP (HTTP).

The app is a thin MCP client: it connects to the MCP server over HTTP,
discovers tools, and lets the agent call them. It does NOT do retrieval itself.

Usage:  uv run streamlit run app.py
"""
import asyncio
import os

import streamlit as st

from ingest import ingest
from agent import create_rag_agent, stream_agent_response

st.title("🤖 Agentic RAG over MCP (HTTP)")

# Initialise session state
if "agent" not in st.session_state:
    st.session_state.agent = None
if "mcp_client" not in st.session_state:
    st.session_state.mcp_client = None

# --- Sidebar: upload + ingest ---
st.sidebar.header("Document Upload")
uploaded_file = st.sidebar.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file:
    os.makedirs("documents", exist_ok=True)
    temp_path = os.path.join("documents", uploaded_file.name)
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    with st.spinner("Ingesting into Qdrant..."):
        ingest(temp_path)

    st.success("Document ingested! Make sure mcp_server.py is running.")

# --- Build the agent once (connects to the MCP server over HTTP) ---
if st.session_state.agent is None:
    with st.spinner("Connecting to MCP server over HTTP..."):
        try:
            agent, client = asyncio.run(create_rag_agent())
            st.session_state.agent = agent
            st.session_state.mcp_client = client
            st.sidebar.success("Connected to MCP server.")
        except Exception as e:
            st.error(f"Could not connect to MCP server: {e}")
            st.info("Make sure `uv run python mcp_server.py` is running first.")

# --- Chat interface ---
st.header("Chat with your Documents")
user_input = st.chat_input("Ask a question about your documents...")

if user_input and st.session_state.agent:
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

        asyncio.run(run_stream())
        response_placeholder.markdown(full_response)

elif user_input and not st.session_state.agent:
    st.warning("Please ensure the MCP server is running and try again.")
