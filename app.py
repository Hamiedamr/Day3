# app.py
import asyncio
import os
import streamlit as st

from ingest import ingest
from agent import create_rag_agent
from streaming import stream_agent_response

st.title("Agentic RAG over MCP (HTTP)")

if "agent" not in st.session_state:
    st.session_state.agent = None

# --- Sidebar: upload + ingest ---
uploaded_file = st.sidebar.file_uploader("Upload a PDF document", type=["pdf"])

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
        st.session_state.agent = asyncio.run(create_rag_agent())

# --- Chat interface ---
user_input = st.chat_input("Ask a question about your document...")

if user_input:
    st.chat_message("user").write(user_input)

    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""

        async def run_stream():
            nonlocal_response = []
            async for token in stream_agent_response(
                st.session_state.agent,
                user_input,
                thread_id="session_001"
            ):
                nonlocal_response.append(token)
                response_placeholder.markdown("".join(nonlocal_response) + "▌")
            return "".join(nonlocal_response)

        full_response = asyncio.run(run_stream())
        response_placeholder.markdown(full_response)