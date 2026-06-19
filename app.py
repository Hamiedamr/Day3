import streamlit as st
import os
from qdrant_client import QdrantClient

# Import your tasks
from task7 import load_and_split_pdf
from task8 import setup_qdrant_collection
from task10 import create_rag_agent
from task11 import stream_agent_response

st.title("Agentic Hybrid RAG Explorer")

if not os.path.exists("data"):
    os.makedirs("data")

# Initialize session states safely
if "agent" not in st.session_state:
    st.session_state.agent = None
if "dense_embedder" not in st.session_state:
    st.session_state.dense_embedder = None
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar for file uploading
uploaded_file = st.sidebar.file_uploader("Upload your training documentation (PDF)", type=["pdf"])

if uploaded_file:
    temp_path = os.path.join("data", uploaded_file.name)
    
    # Only write and process if the agent hasn't been built yet
    if st.session_state.agent is None:
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
            
        with st.spinner("Processing document with native Qdrant..."):
            chunks = load_and_split_pdf(temp_path)
            
            qdrant_client, collection_name, dense_embedder, sparse_embedder = setup_qdrant_collection(chunks)
            
            # Keep trackers alive in memory across chat interactions
            st.session_state.dense_embedder = dense_embedder
            st.session_state.sparse_embedder = sparse_embedder
            
            st.session_state.agent = create_rag_agent(
                qdrant_client,
                collection_name,
                dense_embedder,
                sparse_embedder
            )
        st.sidebar.success("Document processed successfully!")

# Persistent historical chat message render window
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# Chat Interface
user_input = st.chat_input("Ask a question about your document...")

if user_input:
    if st.session_state.agent:
        st.chat_message("user").write(user_input)
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            full_response = ""
            
            for token in stream_agent_response(
                st.session_state.agent,
                user_input,
                thread_id="session_001"
            ):
                full_response += token
                response_placeholder.markdown(full_response + "▌")
            
            response_placeholder.markdown(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})
    else:
        st.warning("Please upload a document first!")