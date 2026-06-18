"""
Agentic RAG System - Complete Solution
Uses: Ollama + Qdrant (native) + FastEmbed + LangChain + LangGraph + Streamlit
"""

import os
import uuid
import streamlit as st

from langchain_pymupdf4llm import PyMuPDF4LLMLoader
from langchain_text_splitters import MarkdownTextSplitter

from qdrant_client import QdrantClient, models
from fastembed import TextEmbedding, SparseTextEmbedding

from langchain_ollama import ChatOllama
from langchain.tools import tool
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

# Task 7: Document Loading & Text Splitting

def load_and_split_pdf(file_path: str):
    """Load a PDF and split it into overlapping markdown chunks."""
    loader = PyMuPDF4LLMLoader(file_path=file_path)
    docs = loader.load()

    splitter = MarkdownTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    return chunks



# Task 8: Native Qdrant Collection + Points

QDRANT_URL        = "http://localhost:6333"
COLLECTION_NAME   = "rag_documents"
DENSE_MODEL       = "jinaai/jina-embeddings-v2-base-en"
SPARSE_MODEL      = "Qdrant/bm25"

def setup_qdrant_collection(chunks):
    """
    Create (or reuse) a Qdrant collection with dense + sparse vectors,
    then upload all chunk points.
    """
    client = QdrantClient(url=QDRANT_URL)

    dense_embedder  = TextEmbedding(model_name=DENSE_MODEL)
    sparse_embedder = SparseTextEmbedding(model_name=SPARSE_MODEL)

    sample_embedding = list(dense_embedder.embed(["sample text"]))[0]
    vector_size = len(sample_embedding)

    if not client.collection_exists(collection_name=COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config={
                "dense": models.VectorParams(
                    size=vector_size,
                    distance=models.Distance.COSINE,
                )
            },
            sparse_vectors_config={
                "sparse": models.SparseVectorParams(
                    index=models.SparseIndexParams(on_disk=False)
                )
            },
        )

    texts = [chunk.page_content for chunk in chunks]
    dense_vectors = list(dense_embedder.embed(texts))
    sparse_vectors = list(sparse_embedder.embed(texts))

    points = []
    for idx, (chunk, dense_vec, sparse_vec) in enumerate(
        zip(chunks, dense_vectors, sparse_vectors)
    ):
        sparse_indices = sparse_vec.indices.tolist()
        sparse_values  = sparse_vec.values.tolist()

        point = models.PointStruct(
            id=str(uuid.uuid4()),
            vector={
                "dense": (
                    dense_vec.tolist()
                    if hasattr(dense_vec, "tolist")
                    else list(dense_vec)
                ),
                "sparse": models.SparseVector(
                    indices=sparse_indices,
                    values=sparse_values,
                ),
            },
            payload={
                "page_content": chunk.page_content,
                "metadata": chunk.metadata,
                "chunk_id": idx,
            },
        )
        points.append(point)

    client.upload_points(
        collection_name=COLLECTION_NAME,
        points=points,
        batch_size=64,
        parallel=2,
        max_retries=3,
        wait=False,
    )

    return client, COLLECTION_NAME, dense_embedder, sparse_embedder


# Task 9: Hybrid Search with RRF Fusion

def hybrid_search_rrf(
    client,
    collection_name,
    query_text,
    dense_embedder,
    sparse_embedder,
    limit=5,
):
    """Hybrid search: dense (semantic) + sparse (BM25) fused with RRF."""
    dense_query  = list(dense_embedder.embed([query_text]))[0]
    sparse_query = list(sparse_embedder.embed([query_text]))[0]

    sparse_indices = sparse_query.indices.tolist()
    sparse_values  = sparse_query.values.tolist()

    results = client.query_points(
        collection_name=collection_name,
        prefetch=[
            models.Prefetch(
                query=models.SparseVector(
                    indices=sparse_indices,
                    values=sparse_values,
                ),
                using="sparse",
                limit=20,
            ),
            models.Prefetch(
                query=(
                    dense_query.tolist()
                    if hasattr(dense_query, "tolist")
                    else list(dense_query)
                ),
                using="dense",
                limit=20,
            ),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        with_payload=True,
        limit=limit,
    )

    documents = []
    for point in results.points:
        documents.append(
            {
                "content":  point.payload["page_content"],
                "metadata": point.payload["metadata"],
                "score":    point.score,
                "id":       point.id,
            }
        )
    return documents


# Task 10: Agentic RAG – Agent Creation

def create_rag_agent(qdrant_client, collection_name, dense_embedder, sparse_embedder):
    """Build a ReAct agent that calls the hybrid-search tool when needed."""

    @tool(response_format="content_and_artifact")
    def retrieve_context(query: str):
        """Retrieve relevant documents using hybrid search with RRF fusion.

        Args:
            query: The search query to find relevant documents.
        """
        docs = hybrid_search_rrf(
            client=qdrant_client,
            collection_name=collection_name,
            query_text=query,
            dense_embedder=dense_embedder,
            sparse_embedder=sparse_embedder,
            limit=5,
        )

        serialized = "\n\n".join(
            f"Source: {doc['metadata']}\nContent: {doc['content']}\nScore: {doc['score']}"
            for doc in docs
        )
        return serialized, docs

    system_prompt = """You are a helpful AI assistant with access to a document knowledge base.

Instructions:
- Use the retrieve_context tool whenever you need information from the uploaded documents.
- The retrieval uses hybrid search (semantic + keyword) with RRF fusion for the best results.
- Always cite your sources when using retrieved information.
- If the retrieved context does not contain relevant information, say:
  "I don't have enough information to answer that question."
- You can ask clarifying follow-up questions if the query is unclear.
"""

    llm = ChatOllama(model="qwen3:4b-instruct", temperature=0)

    checkpointer = InMemorySaver()

    agent = create_react_agent(
        model=llm,
        tools=[retrieve_context],
        prompt=system_prompt,          
        checkpointer=checkpointer,
    )

    return agent


# Task 11: Streaming Agent Responses

def stream_agent_response(agent, user_query, thread_id="default"):
    """Stream the agent's response token-by-token using stream_mode='values'."""
    inputs = {"messages": [{"role": "user", "content": user_query}]}
    config = {"configurable": {"thread_id": thread_id}}

    for chunk in agent.stream(inputs, stream_mode="values", config=config):
        latest_message = chunk["messages"][-1]

        if isinstance(latest_message, AIMessage) and latest_message.content:
            yield latest_message.content

        elif hasattr(latest_message, "tool_calls") and latest_message.tool_calls:
            tool_names = [tc["name"] for tc in latest_message.tool_calls]
            yield f"\n Searching: {tool_names}\n"

        elif isinstance(latest_message, ToolMessage):
            yield "\n Context retrieved.\n"


# Task 12: Streamlit UI

st.set_page_config(page_title="Agentic RAG System", page_icon="🤖", layout="wide")
st.title(" Agentic RAG System")
st.caption("Powered by Ollama · Qdrant · FastEmbed · LangGraph")

for key in ["agent", "qdrant_client", "collection_name", "dense_embedder", "sparse_embedder"]:
    if key not in st.session_state:
        st.session_state[key] = None

with st.sidebar:
    st.header(" Upload Document")
    uploaded_file = st.file_uploader(
        "Upload a PDF to add to the knowledge base",
        type=["pdf"],
    )

    if uploaded_file:
        os.makedirs("documents", exist_ok=True)
        temp_path = os.path.join("documents", uploaded_file.name)

        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        with st.spinner(" Processing document – this may take a minute …"):
            chunks = load_and_split_pdf(temp_path)
            st.write(f"Split into **{len(chunks)}** chunks.")

            qdrant_client, collection_name, dense_embedder, sparse_embedder = (
                setup_qdrant_collection(chunks)
            )

            st.session_state.qdrant_client   = qdrant_client
            st.session_state.collection_name = collection_name
            st.session_state.dense_embedder  = dense_embedder
            st.session_state.sparse_embedder = sparse_embedder

            st.session_state.agent = create_rag_agent(
                qdrant_client,
                collection_name,
                dense_embedder,
                sparse_embedder,
            )

        st.success("Document ready! Hybrid search (RRF) is active.")

    st.divider()
    st.info(
        "**How it works**\n\n"
        "1. Upload a PDF\n"
        "2. The system chunks & embeds it\n"
        "3. Ask any question – the agent decides when to search\n"
        "4. Results fuse semantic + keyword search via RRF"
    )

user_input = st.chat_input("Ask a question about your document …")

if user_input:
    if not st.session_state.agent:
        st.warning(" Please upload a PDF document first!")
    else:
        with st.chat_message("user"):
            st.write(user_input)

        with st.chat_message("assistant"):
            placeholder   = st.empty()
            full_response = ""

            for token in stream_agent_response(
                st.session_state.agent,
                user_input,
                thread_id="session_001",
            ):
                full_response += token
                placeholder.markdown(full_response + "▌")

            placeholder.markdown(full_response)