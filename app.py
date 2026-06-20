import os
import uuid
import streamlit as st
from langchain_pymupdf4llm import PyMuPDF4LLMLoader
from langchain_text_splitters import MarkdownTextSplitter
from qdrant_client import QdrantClient, models
from fastembed import TextEmbedding, SparseTextEmbedding
from langchain.agents import create_agent
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langchain.messages import AIMessage, HumanMessage

def load_and_split_pdf(file_path):
    loader = PyMuPDF4LLMLoader(file_path=file_path)
    docs = loader.load()
    splitter = MarkdownTextSplitter(chunk_size=1000, chunk_overlap=100)
    return splitter.split_documents(docs)

def setup_qdrant_collection(chunks):
    client = QdrantClient("http://localhost:6333")
    dense_embedder = TextEmbedding("BAAI/bge-small-en-v1.5")
    sparse_embedder = SparseTextEmbedding("Qdrant/bm25")
    collection_name = "rag"
    sample_embedding = list(dense_embedder.embed(["sample"]))[0]
    vector_size = len(sample_embedding)
    if not client.collection_exists(collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config={
                "dense": models.VectorParams(
                    size=vector_size,
                    distance=models.Distance.COSINE
                )
            },
            sparse_vectors_config={
                "sparse": models.SparseVectorParams(
                    index=models.SparseIndexParams(on_disk=False)
                )
            }
        )
    points = []
    texts = [chunk.page_content for chunk in chunks]
    dense_vectors = list(dense_embedder.embed(texts))
    sparse_vectors = list(sparse_embedder.embed(texts))
    for idx, (chunk, dense_vec, sparse_vec) in enumerate(zip(chunks, dense_vectors, sparse_vectors)):
        points.append(
            models.PointStruct(
                id=str(uuid.uuid4()),
                vector={
                    "dense": dense_vec.tolist() if hasattr(dense_vec, 'tolist') else list(dense_vec),
                    "sparse": models.SparseVector(
                        indices=sparse_vec.indices.tolist(),
                        values=sparse_vec.values.tolist()
                    )
                },
                payload={
                    "page_content": chunk.page_content,
                    "metadata": chunk.metadata,
                    "chunk_id": idx
                }
            )
        )
    client.upload_points(
        collection_name=collection_name,
        points=points,
        batch_size=64,
        parallel=2,
        max_retries=3,
        wait=False
    )
    return client, collection_name, dense_embedder, sparse_embedder

def hybrid_search_rrf(client, collection_name, query_text, dense_embedder, sparse_embedder, limit=5):
    dense_query = list(dense_embedder.embed([query_text]))[0]
    sparse_query = list(sparse_embedder.embed([query_text]))[0]
    sparse_indices = sparse_query.indices.tolist()
    sparse_values = sparse_query.values.tolist()
    results = client.query_points(
        collection_name=collection_name,
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
                query=dense_query.tolist() if hasattr(dense_query, 'tolist') else list(dense_query),
                using="dense",
                limit=20
            )
        ],
        query=models.FusionQuery(
            fusion=models.Fusion.RRF
        ),
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

def create_rag_agent(qdrant_client, collection_name, dense_embedder, sparse_embedder):
    @tool(response_format="content_and_artifact")
    def retrieve_context(query: str):
        """Search documents."""
        docs = hybrid_search_rrf(
            client=qdrant_client,
            collection_name=collection_name,
            query_text=query,
            dense_embedder=dense_embedder,
            sparse_embedder=sparse_embedder,
            limit=5
        )
        serialized = "\n\n".join(
            f"Source: {doc['metadata']}\nContent: {doc['content']}\nScore: {doc['score']}"
            for doc in docs
        )
        return serialized, docs
    
    system_prompt = """You are a helpful AI assistant with access to a document knowledge base.

Instructions:
- Use the retrieve_context tool when you need information from the documents
- The retrieval uses hybrid search (semantic + keyword) with RRF fusion for best results
- Always cite your sources when using retrieved information
- If the retrieved context doesn't contain relevant information, say "I don't have enough information to answer that question"
- You can ask follow-up questions if the query is unclear"""
    
    return create_agent(
        model="ollama:qwen3:4b-instruct",
        tools=[retrieve_context],
        system_prompt=system_prompt,
        checkpointer=InMemorySaver()
    )

def stream_agent_response(agent, user_query, thread_id="default"):
    inputs = {
        "messages": [{"role": "user", "content": user_query}]
    }
    config = {
        "configurable": {
            "thread_id": thread_id
        }
    }
    for chunk in agent.stream(
        inputs,
        stream_mode="values",
        config=config
    ):
        latest_message = chunk["messages"][-1]
        if isinstance(latest_message, AIMessage) and latest_message.content:
            yield latest_message.content
        elif hasattr(latest_message, 'tool_calls') and latest_message.tool_calls:
            yield f"\nSearching: {[tc['name'] for tc in latest_message.tool_calls]}\n"

st.title("Agentic RAG System")

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

uploaded_file = st.sidebar.file_uploader("Upload PDF")

if uploaded_file:
    temp_path = os.path.join("documents", uploaded_file.name)
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    with st.spinner("Processing document..."):
        chunks = load_and_split_pdf(temp_path)
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
    st.success("Document processed. Agent is ready.")

user_input = st.chat_input("Ask a question")

if user_input and st.session_state.agent:
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
elif user_input and not st.session_state.agent:
    st.warning("Please upload a document first.")
