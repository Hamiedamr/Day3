import os
import streamlit as st
from qdrant_client import QdrantClient
from rag_core import load_and_split_pdf, setup_qdrant_collection, create_rag_agent, stream_agent_response

st.title("Agentic RAG Assistant")

if "agent" not in st.session_state:
    st.session_state.agent = None

if st.session_state.agent is None:
    try:
        client = QdrantClient(url="http://localhost:6333")
        if client.collection_exists("rag_demo"):
            info = client.get_collection("rag_demo")
            if info.points_count > 0:
                st.session_state.agent = create_rag_agent()
                st.sidebar.success("Auto-connected to existing Qdrant collection!")
    except Exception as e:
        st.sidebar.error(f"Could not auto-connect: {e}")

uploaded_file = st.sidebar.file_uploader("Upload PDF Document", type=["pdf"])

if uploaded_file:
    temp_path = os.path.join("documents", uploaded_file.name)
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    with st.spinner("Processing document..."):
        chunks = load_and_split_pdf(temp_path)
        qdrant_client, collection_name, dense_embedder, sparse_embedder = setup_qdrant_collection(chunks)
        st.session_state.agent = create_rag_agent()
    st.success("Document processed! Agent is ready.")

user_input = st.chat_input("Ask a question about the document...")

if user_input and st.session_state.agent:
    st.chat_message("user").write(user_input)
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""
        for token in stream_agent_response(st.session_state.agent, user_input, thread_id="session_001"):
            full_response += token
            response_placeholder.markdown(full_response + "▌")
        response_placeholder.markdown(full_response)
elif user_input and not st.session_state.agent:
    st.warning("Please upload a document first!")
