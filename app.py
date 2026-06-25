import asyncio
import os
import streamlit as st
import nest_asyncio

nest_asyncio.apply()

from ingest import ingest
from agent import create_rag_agent, build_mcp_client

st.set_page_config(page_title="Agentic RAG MCP", layout="wide")
st.title("🤖 Agentic RAG over MCP (HTTP)")

if "agent" not in st.session_state:
    st.session_state.agent = None

with st.sidebar:
    st.header("📄 Document Upload")
    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

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
        st.session_state.agent = asyncio.run(create_rag_agent())

user_input = st.chat_input("Ask a question about your document...")

if user_input:
    st.chat_message("user").write(user_input)

    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""

        async def run_stream():
            client = build_mcp_client()
            tools = await client.get_tools()
            
            tool = tools[0]
            result = await tool.ainvoke({"query": user_input})
            
            return result

        result = asyncio.run(run_stream())
        response_placeholder.markdown(result)