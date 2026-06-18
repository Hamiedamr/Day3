import streamlit as st
import os
from pdf_processor import load_and_split_pdf, setup_qdrant_collection
from agent import create_rag_agent, stream_agent_response

st.title("Agentic RAG Chatbot (Hybrid Search)")

if "agent" not in st.session_state:
    st.session_state.agent = None
if "messages" not in st.session_state:
    st.session_state.messages = []

st.sidebar.header("Document Management")
uploaded_file = st.sidebar.file_uploader("Upload your PDF document", type=["pdf"])

if uploaded_file and st.session_state.agent is None:
    os.makedirs("documents", exist_ok=True)
    temp_path = os.path.join("documents", uploaded_file.name)
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    with st.spinner("Processing document..."):
        chunks = load_and_split_pdf(temp_path)
        client, col_name, dense, sparse = setup_qdrant_collection(chunks)
        st.session_state.agent = create_rag_agent(client, col_name, dense, sparse)
    st.success("Document processed! Agent is ready.")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

user_input = st.chat_input("Ask me anything about the document...")

if user_input and st.session_state.agent:
    st.chat_message("user").write(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""

        for token in stream_agent_response(st.session_state.agent, user_input, thread_id="session_001"):
            full_response += token
            response_placeholder.markdown(full_response + "▌")

        response_placeholder.markdown(full_response)
        st.session_state.messages.append({"role": "assistant", "content": full_response})

elif user_input and not st.session_state.agent:
    st.warning("Please upload a document first!")