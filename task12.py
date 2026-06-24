import streamlit as st
import os
from qdrant_client import QdrantClient

# Import your functions
from task7 import load_and_split_pdf
from task8 import setup_qdrant_collection
from task10 import create_rag_agent
from task11 import stream_agent_response

# Create page title
st.title(" Agentic RAG Assistant")

# Initialize session state for agent and Qdrant components
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

# Auto-initialize agent on startup if the Qdrant database is already populated
if st.session_state.agent is None:
    try:
        client = QdrantClient(url="http://localhost:6333")
        if client.collection_exists("javascript_guide"):
            info = client.get_collection("javascript_guide")
            if info.points_count > 0:
                st.session_state.agent = create_rag_agent()
                st.sidebar.success("✅ Auto-connected to existing Qdrant collection!")
    except Exception as e:
        st.sidebar.error(f"Could not auto-connect: {e}")

# Create Sidebar for File Upload
uploaded_file = st.sidebar.file_uploader("Upload PDF Document", type=["pdf"])

if uploaded_file:
    # Save file locally
    temp_path = os.path.join("documents", uploaded_file.name)
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Process and store document
    with st.spinner("Processing document with native Qdrant..."):
        # Load and split PDF
        chunks = load_and_split_pdf(temp_path)
        
        # Setup native Qdrant collection with dense + sparse vectors
        qdrant_client, collection_name, dense_embedder, sparse_embedder = setup_qdrant_collection(chunks)
        
        st.session_state.qdrant_client = qdrant_client
        st.session_state.collection_name = collection_name
        st.session_state.dense_embedder = dense_embedder
        st.session_state.sparse_embedder = sparse_embedder
        
        # Create the agent with native Qdrant components
        st.session_state.agent = create_rag_agent()
    
    st.success("Document processed with hybrid search (RRF fusion)! Agent is ready.")

# Create Chat Interface
user_input = st.chat_input("Ask a question about the document...")

if user_input and st.session_state.agent:
    # Display user message
    st.chat_message("user").write(user_input)
    
    # Display assistant response with streaming
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""
        
        # Stream the response using agent.stream (synchronous — no async needed)
        for token in stream_agent_response(
            st.session_state.agent,
            user_input,
            thread_id="session_001"  # Use consistent thread_id for memory
        ):
            full_response += token
            response_placeholder.markdown(full_response + "▌")
        
        # Final update without cursor
        response_placeholder.markdown(full_response)
        
elif user_input and not st.session_state.agent:
    st.warning("Please upload a document first!")