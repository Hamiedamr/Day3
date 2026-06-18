import streamlit as st
import os
from fastembed import TextEmbedding, SparseTextEmbedding

from task7 import load_and_split_pdf
from task8 import setup_qdrant_collection 
from task10_agent import create_rag_agent
from task11_streaming import stream_agent_response


st.title("Agentic RAG System - Market Insights")

if not os.path.exists("documents"):
    os.makedirs("documents")

@st.cache_resource
def load_embedding_models():
    """Initializes and caches dense and sparse models to prevent memory leaks."""
    dense = TextEmbedding(model_name="jinaai/jina-embeddings-v2-base-en")
    sparse = SparseTextEmbedding(model_name="Qdrant/bm25")
    return dense, sparse

if "agent" not in st.session_state:
    st.session_state.agent = None
if "processed" not in st.session_state:
    st.session_state.processed = False

dense_embedder, sparse_embedder = load_embedding_models()

uploaded_file = st.sidebar.file_uploader("Upload Market Research PDF", type=["pdf"])

if uploaded_file and not st.session_state.processed:
    temp_path = os.path.join("documents", uploaded_file.name)
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
        
    with st.spinner("Processing document with native Qdrant..."):
      
        chunks = load_and_split_pdf(temp_path)
        
        
        setup_qdrant_collection(chunks, dense_embedder, sparse_embedder)
        
      
        st.session_state.agent = create_rag_agent(dense_embedder, sparse_embedder)
        st.session_state.processed = True
        
    st.sidebar.success("Document processed with hybrid search (RRF fusion)! Agent is ready.")

user_input = st.chat_input("Ask a question about the document...")

if user_input:
    if st.session_state.agent:
        st.chat_message("user").write(user_input)
        
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
    else:
        st.warning("Please upload a document first!")