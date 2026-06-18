import streamlit as st
import os

from document_processor import load_and_split_pdf
from vdb_setup import setup_qdrant_collection
from rag_agent import create_rag_agent
from streaming_engine import stream_agent_response

os.makedirs("documents", exist_ok=True)

st.set_page_config(page_title="Agentic Hybrid RAG", page_icon="🤖", layout="wide")
st.title("🤖 Intelligent Agentic RAG System")
st.caption("Powered by Local Qdrant Hybrid Search (Dense + Sparse), RRF Fusion, and LangGraph 2026 Engine")

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
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.sidebar.header(" Knowledge Base Ingestion")

base_dir = os.path.dirname(os.path.abspath(__file__))
default_pdf_path = os.path.join(base_dir, "documents", "test.pdf")

if os.path.exists(default_pdf_path):
    if st.session_state.agent is None:
        with st.spinner("Processing local 'test.pdf' with native Qdrant (Splitting & Embedding)..."):
            try:
                chunks = load_and_split_pdf(default_pdf_path)
                st.sidebar.info(f"📄 Generated {len(chunks)} Markdown document chunks.")
                
                qdrant_client, collection_name, dense_embedder, sparse_embedder = setup_qdrant_collection(chunks)
                
                st.session_state.qdrant_client = qdrant_client
                st.session_state.collection_name = collection_name
                st.session_state.dense_embedder = dense_embedder
                st.session_state.sparse_embedder = sparse_embedder
                
                st.session_state.agent = create_rag_agent(
                    qdrant_client,
                    collection_name,
                    dense_embedder,
                    sparse_embedder
                )
                st.sidebar.success(" Hybrid indexing finished. Agent is active!")
            except Exception as e:
                st.sidebar.error(f"Failed to index document: {str(e)}")
else:
    st.sidebar.warning(f" 'test.pdf' not found in documents/ folder. Please make sure it exists.")

for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

user_input = st.chat_input("Ask a question about your uploaded document...")

if user_input:
    if st.session_state.agent:
        st.chat_message("user").write(user_input)
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            full_response = ""
            
            for token in stream_agent_response(
                st.session_state.agent,
                user_input,
                thread_id="session_streamlit_001"
            ):
                full_response += token
                response_placeholder.markdown(full_response + "▌")
            
            response_placeholder.markdown(full_response)
            
            st.session_state.chat_history.append({"role": "assistant", "content": full_response})
            
    else:
        st.warning(" The agent is not initialized yet. Please make sure 'test.pdf' is inside the documents folder.")