import os
import streamlit as st
from langgraph.checkpoint.memory import InMemorySaver
from rag_core import load_and_split_pdf, setup_qdrant_collection, create_rag_agent, stream_agent_response


def main():
    st.set_page_config(page_title="Agentic RAG Assistant", page_icon="⚡", layout="wide")



    st.markdown("""

        <h1>Agentic RAG Assistant</h1>
    """, unsafe_allow_html=True)

    if "agent" not in st.session_state:
        st.session_state.agent = None
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "checkpointer" not in st.session_state:
        st.session_state.checkpointer = InMemorySaver()

    uploaded_file = st.sidebar.file_uploader("Upload PDF Document", type=["pdf"])

    if uploaded_file:
        os.makedirs("documents", exist_ok=True)
        temp_path = os.path.join("documents", uploaded_file.name)
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        if not st.session_state.agent:
            with st.spinner("Processing document..."):
                chunks = load_and_split_pdf(temp_path)
                qdrant_client, collection_name, dense_embedder, sparse_embedder = setup_qdrant_collection(chunks)
                st.session_state.agent = create_rag_agent(
                    qdrant_client, collection_name, dense_embedder, sparse_embedder,
                    checkpointer=st.session_state.checkpointer
                )
            st.sidebar.success("Document processed! Agent is ready.")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    user_input = st.chat_input("Ask a question about the uploaded document...")

    if user_input and st.session_state.agent:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

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


if __name__ == "__main__":
    main()
