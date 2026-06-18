import streamlit as st
import os

from task7 import load_and_split_pdf
from task8 import setup_qdrant_collection
from task9 import hybrid_search_rrf
from langchain.chat_models import init_chat_model

# TODO: Create page title
st.title("📚 Agentic RAG System with Hybrid Search")

# TODO: Initialize session state for agent and Qdrant components
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

# TODO: Create Sidebar for File Upload
uploaded_file = st.sidebar.file_uploader("Upload a PDF document to query", type=["pdf"])

if uploaded_file:
    # TODO: Save file locally
    temp_path = os.path.join("temp_uploads", uploaded_file.name)
    os.makedirs("temp_uploads", exist_ok=True)
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # TODO: Process and store document
    with st.spinner("Processing document with native Qdrant..."):
        # Load and split PDF
        chunks = load_and_split_pdf(temp_path)

        # Setup native Qdrant collection with dense + sparse vectors
        qdrant_client, collection_name, dense_embedder, sparse_embedder = setup_qdrant_collection(
            chunks
        )

        st.session_state.qdrant_client = qdrant_client
        st.session_state.collection_name = collection_name
        st.session_state.dense_embedder = dense_embedder
        st.session_state.sparse_embedder = sparse_embedder

        # Store the model (no agent tool-calling — retrieve-then-read instead)
        st.session_state.agent = True  # Flag that we're ready

    st.success("Document processed with hybrid search (RRF fusion)! Agent is ready.")

# TODO: Create Chat Interface
user_input = st.chat_input("Ask a question about your document...")

if user_input and st.session_state.qdrant_client:
    # TODO: Display user message
    st.chat_message("user").write(user_input)

    # TODO: Display assistant response with streaming
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""

        # Retrieve relevant context using hybrid search with RRF fusion
        docs = hybrid_search_rrf(
            st.session_state.qdrant_client,
            st.session_state.collection_name,
            user_input,
            st.session_state.dense_embedder,
            st.session_state.sparse_embedder,
            limit=5,
        )

        context = "\n\n".join(
            f"Source: {doc['metadata']}\nContent: {doc['content']}"
            for doc in docs
        )

        model = init_chat_model("ollama:qwen3:4b-instruct")

        system_prompt = f"""You are a helpful AI assistant with access to a document knowledge base.

Retrieved context from the user's document:
{context}

Instructions:
- Answer the user's question based ONLY on the retrieved context above.
- If the context doesn't contain relevant information, say "I don't have enough information to answer that question."
- Cite the specific source details from the context.

User question: {user_input}
"""

        # Stream the response token by token
        for chunk in model.stream([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input},
        ]):
            if chunk.content:
                full_response += chunk.content
                response_placeholder.markdown(full_response + "▌")

        # Final update without cursor
        response_placeholder.markdown(full_response)

elif user_input and not st.session_state.qdrant_client:
    st.warning("Please upload a document first!")