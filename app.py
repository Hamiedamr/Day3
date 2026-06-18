import importlib

import streamlit as st
from task7 import load_and_split_pdf 
from task8 import setup_qdrant_collection
import task10
import task11

task10 = importlib.reload(task10)
task11 = importlib.reload(task11)

st.set_page_config(page_title="Agentic RAG System", layout="centered")
st.title("🤖 Agentic RAG System")

st.sidebar.header("Upload Document")
uploaded_file = st.sidebar.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    if st.button("Process Document"):
        with st.spinner("Processing and Indexing..."):
            chunks = load_and_split_pdf("temp.pdf")
            client, col_name, dense_model, sparse_model = setup_qdrant_collection(chunks)
            
            st.session_state.agent = task10.create_rag_agent(
                client, col_name, dense_model, sparse_model
            )
            st.success("Agent is ready!")

if "agent" in st.session_state:
    st.subheader("Chat with your Document")
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask a question..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Agent is thinking..."):
                response_placeholder = st.empty()
                full_response = ""

                for token in task11.stream_agent_response(
                    st.session_state.agent,
                    prompt,
                    thread_id="session_1"
                ):
                    full_response += token
                    response_placeholder.markdown(full_response + "▌")

                response_placeholder.markdown(full_response)
                st.session_state.messages.append(
                    {"role": "assistant", "content": full_response}
                )

elif not uploaded_file:
    st.info("Please upload a PDF to get started.")
