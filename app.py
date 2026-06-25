import asyncio
import os
import streamlit as st
from langgraph.checkpoint.memory import InMemorySaver

from ingest import ingest
from agent import create_rag_agent, stream_agent_response

st.title("Agentic RAG over MCP")

if "checkpointer" not in st.session_state:
    st.session_state.checkpointer = InMemorySaver()

uploaded_file = st.sidebar.file_uploader("Upload PDF")

if uploaded_file:
    os.makedirs("documents", exist_ok=True)
    temp_path = os.path.join("documents", uploaded_file.name)
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    with st.spinner("Ingesting into Qdrant..."):
        ingest(temp_path)
    st.success("Document ingested.")

user_input = st.chat_input("Ask a question")

if user_input:
    st.chat_message("user").write(user_input)
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = [""]
        async def run_stream():
            agent = await create_rag_agent(st.session_state.checkpointer)
            async for token in stream_agent_response(
                agent,
                user_input,
                thread_id="session_001"
            ):
                full_response[0] += token
                response_placeholder.markdown(full_response[0] + "▌")
        loop = asyncio.new_event_loop()
        loop.run_until_complete(run_stream())
        loop.close()
        response_placeholder.markdown(full_response[0])
