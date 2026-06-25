"""
Agentic RAG Application — online search & UI stage.

Usage:  uv run streamlit run app.py
"""
import os

import streamlit as st
from qdrant_client import models
from langchain.agents import create_agent
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langchain.messages import AIMessage

import ingest  # module reference — reads ingest._search_* live (not import-time snapshot)

COLLECTION_NAME = "my_agentic_rag_collection"
LLM_MODEL = "ollama:qwen3:4b-instruct"


# ── Task 9: Hybrid Search with RRF Fusion ─────────────────────

def hybrid_search_rrf(query_text, limit=5):
    """Perform hybrid search using dense + sparse vectors with RRF fusion."""
    dense_query = list(ingest._search_dense.embed([query_text]))[0]
    sparse_query = list(ingest._search_sparse.embed([query_text]))[0]

    sparse_indices = sparse_query.indices.tolist()
    sparse_values = sparse_query.values.tolist()

    results = ingest._search_client.query_points(
        collection_name=ingest._search_collection,
        prefetch=[
            models.Prefetch(
                query=models.SparseVector(
                    indices=sparse_indices,
                    values=sparse_values
                ),
                using="sparse",
                limit=20
            ),
            models.Prefetch(
                query=dense_query.tolist() if hasattr(dense_query, 'tolist')
                       else list(dense_query),
                using="dense",
                limit=20
            )
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        with_payload=True,
        limit=limit
    )

    documents = []
    for point in results.points:
        documents.append({
            "content": point.payload["page_content"],
            "metadata": point.payload["metadata"],
            "score": point.score,
            "id": point.id
        })
    return documents


# ── Task 10: Create the Agentic RAG Agent ─────────────────────

def create_rag_agent():
    """Create a ReAct agent with a hybrid-search retrieval tool."""

    @tool(response_format="content_and_artifact")
    def retrieve_context(query: str):
        """Retrieve relevant documents using hybrid search with RRF fusion.

        Args:
            query: The search query to find relevant documents
        """
        docs = hybrid_search_rrf(query_text=query, limit=5)
        serialized = "\n\n".join(
            f"Source: {doc['metadata']}\n"
            f"Content: {doc['content']}\n"
            f"Score: {doc['score']}"
            for doc in docs
        )
        return serialized, docs

    system_prompt = """You are an AI assistant with access to a document knowledge base.

CRITICAL RULE: You MUST call retrieve_context for EVERY user question that asks about any topic or fact. The only exceptions are pure greetings (like "hi" or "hello") or questions about yourself.

Instructions:
- ALWAYS use the retrieve_context tool first — even if you think you know the answer
- Search using the user's exact question as the query
- If the retrieved documents contain relevant information, base your answer on them and cite sources
- If the retrieved documents do NOT contain relevant information, say "The uploaded documents don't cover this topic"
- Never guess or answer from your own knowledge about document-related topics"""


    agent = create_agent(
        model=LLM_MODEL,
        tools=[retrieve_context],
        system_prompt=system_prompt,
        checkpointer=InMemorySaver(),
    )
    return agent


# ── Task 11: Streaming with Agent ─────────────────────────────

def stream_agent_response(agent, user_query, thread_id="default"):
    """Stream the agent response with stream_mode='values'.

    Yields content tokens as the agent thinks, retrieves, and answers.
    Trails with a source indicator so you know how the answer was produced.
    """
    inputs = {"messages": [{"role": "user", "content": user_query}]}
    config = {"configurable": {"thread_id": thread_id}}

    searched = False

    for chunk in agent.stream(inputs, stream_mode="values", config=config):
        latest_message = chunk["messages"][-1]
        if isinstance(latest_message, AIMessage) and latest_message.content:
            yield latest_message.content
        elif hasattr(latest_message, 'tool_calls') and latest_message.tool_calls:
            searched = True
            yield f"\n🔍 Searching documents...\n"

    yield "\n\n---"
    if searched:
        yield "\n*📄 Answered using retrieved documents*"
    else:
        yield "\n*🧠 Answered from general knowledge*"


# ── Task 12: Streamlit UI ─────────────────────────────────────

from qdrant_client import QdrantClient
from fastembed import TextEmbedding, SparseTextEmbedding

st.set_page_config(page_title="Agentic RAG", layout="wide")
st.title("Agentic RAG with Hybrid Search")

# Initialise session state
if "bootstrapped" not in st.session_state:
    st.session_state.bootstrapped = False
    st.session_state.agent = None
    st.session_state.collection_name = COLLECTION_NAME

# Auto-bootstrap: detect pre-ingested data, wire up without upload
if not st.session_state.bootstrapped:
    client = QdrantClient(url="http://localhost:6333")
    if client.collection_exists(COLLECTION_NAME):
        count = client.count(COLLECTION_NAME).count
        if count > 0:
            embedders_status = st.empty()
            embedders_status.info(f"Loading embedders for {count} pre-indexed chunks...")
            ingest._search_client = client
            ingest._search_collection = COLLECTION_NAME
            ingest._search_dense = TextEmbedding(model_name=ingest.DENSE_MODEL)
            ingest._search_sparse = SparseTextEmbedding(model_name=ingest.SPARSE_MODEL)
            st.session_state.agent = create_rag_agent()
            st.session_state.bootstrapped = True
            embedders_status.empty()
    else:
        client.close()

# Sidebar — file upload (for new docs or first-time setup)
st.sidebar.header("Document Upload")
if st.session_state.agent:
    st.sidebar.success("Agent ready — chat below.")
uploaded_file = st.sidebar.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file:
    temp_dir = "/tmp/rag_uploads"
    os.makedirs(temp_dir, exist_ok=True)
    temp_path = os.path.join(temp_dir, uploaded_file.name)
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    with st.spinner("Processing document with native Qdrant (dense + sparse)..."):
        chunks = ingest.load_and_split_pdf(temp_path)
        q_client, col_name, d_emb, s_emb = ingest.setup_qdrant_collection(chunks)

        st.session_state.agent = create_rag_agent()
        st.session_state.bootstrapped = True

    st.sidebar.success(f"Ready! {len(chunks)} chunks indexed.")

# Chat interface
st.header("Chat with your Documents")
user_input = st.chat_input("Ask a question about your documents...")

if user_input and st.session_state.agent:
    st.chat_message("user").write(user_input)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_response = ""
        for token in stream_agent_response(st.session_state.agent, user_input):
            full_response += token
            placeholder.markdown(full_response + "▌")
        placeholder.markdown(full_response)

elif user_input and not st.session_state.agent:
    st.warning("Please upload a document first!")
