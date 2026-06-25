"""Monolithic Agentic RAG System.

A local, privacy-first AI application that uses an intelligent agent
to decide when to retrieve documents and answer questions about them.
"""

from langchain_pymupdf4llm import PyMuPDF4LLMLoader
from langchain_text_splitters import MarkdownTextSplitter
from qdrant_client import QdrantClient, models
from fastembed import TextEmbedding, SparseTextEmbedding
from langchain.agents import create_agent  # 2026 syntax
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver  # For conversation memory
import uuid
import os
import streamlit as st


def load_and_split_pdf(file_path):
    """Load a PDF and split it into markdown chunks."""
    # Initialize PyMuPDF4LLMLoader with the given file path
    loader = PyMuPDF4LLMLoader(file_path=file_path)

    # Load the document
    docs = loader.load()

    # Initialize MarkdownTextSplitter with appropriate chunk settings
    splitter = MarkdownTextSplitter(chunk_size=1000, chunk_overlap=200)

    # Split the documents into chunks
    chunks = splitter.split_documents(docs)

    return chunks


def setup_qdrant_collection(chunks):
    """Setup Qdrant collection using native client with dense + sparse vectors.

    Uses FastEmbed for generating embeddings locally.
    """
    # Initialize Native Qdrant Client
    client = QdrantClient(url="http://localhost:6333")

    # Initialize embedding models (FastEmbed)
    dense_embedding_model = "jinaai/jina-embeddings-v2-base-en"
    sparse_embedding_model = "Qdrant/bm25"

    dense_embedder = TextEmbedding(model_name=dense_embedding_model)
    sparse_embedder = SparseTextEmbedding(model_name=sparse_embedding_model)

    # Create collection with explicit vector configurations
    collection_name = "documents"

    # Get embedding dimensions by encoding a sample text
    sample_embedding = list(dense_embedder.embed(["sample text"]))[0]
    vector_size = len(sample_embedding)

    # Create collection if it doesn't exist
    if not client.collection_exists(collection_name=collection_name):
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

    # Prepare points for upload with dense + sparse vectors
    points = []
    texts = [chunk.page_content for chunk in chunks]

    # Generate embeddings in batches
    dense_vectors = list(dense_embedder.embed(texts))
    sparse_vectors = list(sparse_embedder.embed(texts))

    for idx, (chunk, dense_vec, sparse_vec) in enumerate(zip(chunks, dense_vectors, sparse_vectors)):
        # Convert sparse vector to Qdrant format
        # FastEmbed returns sparse vectors with .indices and .values attributes
        sparse_indices = sparse_vec.indices.tolist()
        sparse_values = sparse_vec.values.tolist()

        point = models.PointStruct(
            id=str(uuid.uuid4()),  # or use idx for integer IDs
            vector={
                "dense": dense_vec.tolist() if hasattr(dense_vec, 'tolist') else list(dense_vec),
                "sparse": models.SparseVector(
                    indices=sparse_indices,
                    values=sparse_values
                )
            },
            payload={
                "page_content": chunk.page_content,
                "metadata": chunk.metadata,
                "chunk_id": idx
            }
        )
        points.append(point)

    # Upload points using native upload_points (with parallelization)
    client.upload_points(
        collection_name=collection_name,
        points=points,
        batch_size=64,      # e.g., 64 (default)
        parallel=2,         # e.g., 2 workers
        max_retries=3,      # e.g., 3
        wait=False           # Async mode for better performance
    )

    return client, collection_name, dense_embedder, sparse_embedder


def hybrid_search_rrf(client, collection_name, query_text, dense_embedder, sparse_embedder, limit=5):
    """Perform hybrid search using dense + sparse vectors with RRF fusion.

    Uses the native Qdrant Query API with prefetch and RRF fusion.
    """
    # Generate embeddings for the query
    dense_query = list(dense_embedder.embed([query_text]))[0]
    sparse_query = list(sparse_embedder.embed([query_text]))[0]

    # Convert sparse vector
    sparse_indices = sparse_query.indices.tolist()
    sparse_values = sparse_query.values.tolist()

    # Perform hybrid search with RRF fusion using prefetch
    results = client.query_points(
        collection_name=collection_name,
        prefetch=[
            # Sparse vector prefetch (keyword search)
            models.Prefetch(
                query=models.SparseVector(
                    indices=sparse_indices,
                    values=sparse_values
                ),
                using="sparse",
                limit=20  # Retrieve top 20 from sparse search
            ),
            # Dense vector prefetch (semantic search)
            models.Prefetch(
                query=dense_query.tolist() if hasattr(dense_query, 'tolist') else list(dense_query),
                using="dense",
                limit=20  # Retrieve top 20 from dense search
            )
        ],
        # Apply RRF fusion to combine results
        query=models.FusionQuery(
            fusion=models.Fusion.RRF  # Reciprocal Rank Fusion
        ),
        with_payload=True,
        limit=limit  # Final number of results to return
    )

    # Extract and return the documents
    documents = []
    for point in results.points:
        doc = {
            "content": point.payload["page_content"],
            "metadata": point.payload["metadata"],
            "score": point.score,
            "id": point.id
        }
        documents.append(doc)

    return documents


def create_rag_agent(qdrant_client, collection_name, dense_embedder, sparse_embedder):
    """Create an agentic RAG agent with native Qdrant hybrid search tool."""

    # Define the retrieval tool using native Qdrant hybrid search
    @tool(response_format="content_and_artifact")  # Use "content_and_artifact" for best results
    def retrieve_context(query: str):
        """Retrieve relevant documents using hybrid search with RRF fusion.

        Args:
            query: The search query to find relevant documents
        """
        # Call the hybrid_search_rrf function
        docs = hybrid_search_rrf(
            client=qdrant_client,
            collection_name=collection_name,
            query_text=query,
            dense_embedder=dense_embedder,
            sparse_embedder=sparse_embedder,
            limit=5
        )

        # Format the retrieved documents
        serialized = "\n\n".join(
            f"Source: {doc['metadata']}\nContent: {doc['content']}\nScore: {doc['score']}"
            for doc in docs
        )
        return serialized, docs

    # Define the system prompt
    system_prompt = """
    You are a helpful AI assistant with access to a document knowledge base.
    You have NO prior knowledge of the uploaded document — you must use the retrieve_context tool to answer any question about it.

    ## TOOL USAGE RULES (Critical)

    - When the user says retrieve, search, find, look up, list, show, or get information from the document → ALWAYS call retrieve_context. Do NOT try to answer from memory.
    - If the user asks "what are the tasks", "list the steps", "what does the lab contain", or any question about the document's structure or content → ALWAYS call retrieve_context to find the relevant sections.
    - The retrieval uses hybrid search (semantic + keyword) with RRF fusion for best results.

    ## WHEN SEARCH RETURNS NO RESULTS

    If retrieve_context returns "No relevant documents found." or an empty result:
    1. Try 2-3 different search queries with different phrasing. For example, if "lab tasks" returns nothing, try "Task 1", "## Task", or "step by step".
    2. Use the retrieve_context tool with queries that match the document's likely structure (headings, numbered items, keywords).
    3. Only after 3 failed attempts, say "I don't have enough information to answer that question".

    ## OUTPUT RULES

    - Always cite your sources when using retrieved information.
    - When listing structured content (tasks, steps, items), present them in a clear numbered or bulleted format.
    - You can ask follow-up questions if the query is unclear.
    """

    # Create the agent using create_agent (2026 syntax)
    # The model string "ollama:<model>" is used directly — no separate ChatOllama import needed
    agent = create_agent(
        model="ollama:qwen3:4b-instruct",  # provider:model string
        tools=[retrieve_context],            # List containing the retrieve_context tool
        system_prompt=system_prompt,         # System instructions
        checkpointer=InMemorySaver(),        # InMemorySaver() for conversation persistence
    )

    return agent


def stream_agent_response(agent, user_query, thread_id="default"):
    """Stream the agent response with stream_mode='values' (2026 recommended approach).

    Args:
        agent: The created agent
        user_query: The user's question
        thread_id: Conversation thread ID for persistence
    """
    from langchain.messages import AIMessage, HumanMessage

    # Prepare the input messages (2026 format)
    inputs = {
        "messages": [{"role": "user", "content": user_query}]
    }

    # Prepare config with thread_id for conversation memory
    config = {
        "configurable": {
            "thread_id": thread_id  # Required for checkpointer to work
        }
    }

    # Stream with stream_mode="values" (2026 recommended approach)
    # Each chunk contains the full state at that point
    for chunk in agent.stream(
        inputs,
        stream_mode="values",     # "values" for full state at each step
        config=config
    ):
        # Access the latest message from the state
        latest_message = chunk["messages"][-1]

        if isinstance(latest_message, AIMessage) and latest_message.content:
            yield latest_message.content
        elif hasattr(latest_message, 'tool_calls') and latest_message.tool_calls:
            # Agent is calling a tool
            yield f"\n🔍 Searching: {[tc['name'] for tc in latest_message.tool_calls]}\n"


# --- Streamlit UI (Task 12) ---
def main():
    """Build the complete Streamlit interface for the Agentic RAG system."""

    # Create page title
    st.title("📚 Agentic RAG System")

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


    try:
        qdrant_client = QdrantClient(url="http://localhost:6333")
        if qdrant_client.collection_exists("documents"):
            collection_info = qdrant_client.count(collection_name="documents")
            if collection_info.count > 0:
                dense_embedder = TextEmbedding(model_name="jinaai/jina-embeddings-v2-base-en")
                sparse_embedder = SparseTextEmbedding(model_name="Qdrant/bm25")
                st.session_state.qdrant_client = qdrant_client
                st.session_state.collection_name = "documents"
                st.session_state.dense_embedder = dense_embedder
                st.session_state.sparse_embedder = sparse_embedder
                st.session_state.agent = create_rag_agent(
                    qdrant_client, "documents", dense_embedder, sparse_embedder
                )
                st.info(f"Loaded existing database with {collection_info.count} document chunks. Upload a PDF to add more.")
    except Exception:
        pass  # Qdrant not running — fall through to upload prompt

    # Create Sidebar for File Upload
    uploaded_file = st.sidebar.file_uploader("Upload a PDF document")

    if uploaded_file:
        # Save file locally
        temp_path = os.path.join("documents", uploaded_file.name)
        os.makedirs("documents", exist_ok=True)
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
            st.session_state.agent = create_rag_agent(
                qdrant_client,
                collection_name,
                dense_embedder,
                sparse_embedder
            )

        st.success("Document processed with hybrid search (RRF fusion)! Agent is ready.")

    # Create Chat Interface
    user_input = st.chat_input("Ask a question about your document")

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
                thread_id="session_002"  # Use consistent thread_id for memory
            ):
                full_response += token
                response_placeholder.markdown(full_response + "▌")

            # Final update without cursor
            response_placeholder.markdown(full_response)

    elif user_input and not st.session_state.agent:
        st.warning("Please upload a document first!")


if __name__ == "__main__":
    main()
