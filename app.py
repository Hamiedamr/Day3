import asyncio
import os
import streamlit as st

from ingest import ingest
from agent import create_rag_agent, stream_agent_response


st.set_page_config(
    page_title="Agentic RAG over MCP",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Agentic RAG over MCP HTTP")


def run_async(coro):
    """
    Helper to run async functions from Streamlit.
    """
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    if loop.is_running():
        new_loop = asyncio.new_event_loop()
        return new_loop.run_until_complete(coro)

    return loop.run_until_complete(coro)


if "agent" not in st.session_state:
    st.session_state.agent = None

if "messages" not in st.session_state:
    st.session_state.messages = []


uploaded_file = st.sidebar.file_uploader(
    "Upload a PDF document",
    type=["pdf"]
)

if uploaded_file:
    os.makedirs("documents", exist_ok=True)

    temp_path = os.path.join("documents", uploaded_file.name)

    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    with st.spinner("Ingesting into Qdrant..."):
        ingest(temp_path)

    st.sidebar.success("Document ingested successfully!")


if st.session_state.agent is None:
    with st.spinner("Connecting to MCP server over HTTP..."):
        st.session_state.agent = run_async(create_rag_agent())

    st.success("Connected to MCP server.")


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


user_input = st.chat_input("Ask a question about your uploaded documents...")

if user_input:
    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""

        async def run_stream():
            nonlocal_full_response = ""

            async for token in stream_agent_response(
                st.session_state.agent,
                user_input,
                thread_id="session_001"
            ):
                nonlocal_full_response += token
                response_placeholder.markdown(nonlocal_full_response + "▌")

            return nonlocal_full_response

        full_response = run_async(run_stream())

        response_placeholder.markdown(full_response)

    st.session_state.messages.append({
        "role": "assistant",
        "content": full_response
    })





