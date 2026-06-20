import streamlit as st
import os
from ingestion import load_and_split_pdf, setup_qdrant_collection
from agent import create_rag_agent, stream_agent_response

def main():
    st.title("agentic rag lab")  
    if "agent" not in st.session_state:
        st.session_state.agent = None
    uploaded_file = st.sidebar.file_uploader("upload pdf", type=["pdf"]) 
    if uploaded_file:
        if not os.path.exists("documents"):
            os.makedirs("documents")
        path = os.path.join("documents", uploaded_file.name)
        with open(path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # process in spinner
        with st.spinner("processing doc..."):
            # load and split
            chunks = load_and_split_pdf(path)
            client, name, dense, sparse = setup_qdrant_collection(chunks)
            st.session_state.agent = create_rag_agent(client, name, dense, sparse)
        
        st.success("agent is ready!")

    user_input = st.chat_input("ask anything")
    
    if user_input and st.session_state.agent:
        # show user msg
        st.chat_message("user").write(user_input)
        
        with st.chat_message("assistant"):
            placeholder = st.empty()
            full_res = ""
            
            # stream tokens
            for token in stream_agent_response(st.session_state.agent, user_input):
                full_res += token
                placeholder.markdown(full_res + "▌")
            
            placeholder.markdown(full_res)
            
    elif user_input and not st.session_state.agent:
        st.warning("upload a doc first")

if __name__ == "__main__":
    main()
