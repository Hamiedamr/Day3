import os

import streamlit as st

from rag import (
    load_and_split_pdf,
    setup_qdrant_collection,
    create_rag_agent,
    stream_agent_response,
)

st.title("📚 Agentic RAG System")

if "agent" not in st.session_state:
    st.session_state.agent = None
if "qdrant_client" not in st.session_state:
    st.session_state.qdrant_client = None
if "collection_name" not in st.session_state:
    st.session_state.collection_name = None
if "dense_embedder" not in st.session_state:
    st.session_state.dense_embedder = None
if "sparse_embedder" not in st.session_state:
    st.session_state.sparse_embedder = None

uploaded_file = st.sidebar.file_uploader("Upload a PDF document", type=["pdf"])

if uploaded_file and st.session_state.get("processed_file") != uploaded_file.name:
    temp_path = os.path.join("documents", uploaded_file.name)
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    with st.spinner("Processing document with native Qdrant..."):
        chunks = load_and_split_pdf(temp_path)

        qdrant_client, collection_name, dense_embedder, sparse_embedder = (
            setup_qdrant_collection(chunks)
        )

        st.session_state.qdrant_client = qdrant_client
        st.session_state.collection_name = collection_name
        st.session_state.dense_embedder = dense_embedder
        st.session_state.sparse_embedder = sparse_embedder

        st.session_state.agent = create_rag_agent(
            qdrant_client, collection_name, dense_embedder, sparse_embedder
        )

    st.success("Document processed with hybrid search (RRF fusion)! Agent is ready.")
    st.session_state.processed_file = uploaded_file.name

user_input = st.chat_input("Ask a question about your document...")

if user_input and st.session_state.agent:
    st.chat_message("user").write(user_input)

    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""

        for token in stream_agent_response(
            st.session_state.agent,
            user_input,
            thread_id="session_001",
        ):
            full_response += token
            response_placeholder.markdown(full_response + "▌")

        response_placeholder.markdown(full_response)

elif user_input and not st.session_state.agent:
    st.warning("Please upload a document first!")
