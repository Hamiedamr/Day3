
from langchain_pymupdf4llm import PyMuPDF4LLMLoader
from langchain_text_splitters import MarkdownTextSplitter
from qdrant_client import QdrantClient, models
from fastembed import TextEmbedding, SparseTextEmbedding
from langchain.agents import create_agent  # 2026 syntax
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver  # For conversation memory
import uuid
import streamlit as st
import os
# Task 7: Document Loading & Text Splitting
def load_and_split_pdf(file_path):
    # Initialize PyMuPDF4LLMLoader with the given file path
    loader = PyMuPDF4LLMLoader(file_path=file_path)
    
    # Load the document
    docs = loader.load()
    
    # Initialize MarkdownTextSplitter with appropriate chunk settings
    splitter = MarkdownTextSplitter(chunk_size=1000, chunk_overlap=200)
    
    # Split the documents into chunks
    chunks = splitter.split_documents(docs)
    
    return chunks

# Task 8: Native Qdrant Setup - Collection Creation & Points Insertion
def setup_qdrant_collection(chunks):
    """Setup Qdrant collection using native client with dense + sparse vectors.
    
    Uses FastEmbed for generating embeddings locally.
    """
    # Initialize Native Qdrant Client
    client = QdrantClient(url="http://localhost:6333")
    
    # Initialize embedding models (FastEmbed)
    dense_embedding_model = "BAAI/bge-small-en-v1.5"  # e.g., "jinaai/jina-embeddings-v2-base-en"
    sparse_embedding_model = "Qdrant/bm25"  # e.g., "Qdrant/bm25" or "prithivida/Splade_PP_en_v1"
    
    dense_embedder = TextEmbedding(model_name=dense_embedding_model)
    sparse_embedder = SparseTextEmbedding(model_name=sparse_embedding_model)
    
    # Create collection with explicit vector configurations
    collection_name = "javascript_guide"
    
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
    
    # Only upload if the collection is empty
    collection_info = client.get_collection(collection_name)
    if collection_info.points_count == 0:
        print("Collection is empty. Generating embeddings and uploading points...")
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
            parallel=2,        # e.g., 2 workers
            max_retries=3,     # e.g., 3
            wait=True          # Sync mode here to ensure points are fully uploaded before checking counts
        )
    else:
        print(f"Collection already has {collection_info.points_count} points. Skipping upload.")
    
    return client, collection_name, dense_embedder, sparse_embedder

# Task 9: Native Hybrid Search with RRF Fusion
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

# Task 10: Create the Agentic RAG Agent
def create_rag_agent(qdrant_client, collection_name, dense_embedder, sparse_embedder):
    """Create an agentic RAG agent with native Qdrant hybrid search tool."""
    
    # Define the retrieval tool using native Qdrant hybrid search
    @tool(response_format="content_and_artifact")  # Use "content_and_artifact" for best results
    def retrieve_context(query: str) -> tuple[str, list]:
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
    system_prompt = """You are a specialized search agent.
CRITICAL: You do not know anything about JavaScript by default. You MUST use the retrieve_context tool to search for information before you can answer ANY user question. Do not answer from your own knowledge. Always query the retrieve_context tool first."""
    
    # Create the agent using create_agent (2026 syntax)
    # The model string "ollama:<model>" is used directly — no separate ChatOllama import needed
    agent = create_agent(
        model="ollama:qwen3:4b-instruct",           # e.g., "ollama:qwen3:4b-instruct" (provider:model string)
        tools=[retrieve_context],           # List containing the retrieve_context tool
        system_prompt=system_prompt,   # System instructions
        checkpointer=InMemorySaver(),    # InMemorySaver() for conversation persistence
    )
    
    return agent

# Task 11: Streaming with Agent (2026 Recommended)
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

# Task 12: Build the Streamlit UI with Streaming
def main():
    # Page setup
    st.set_page_config(page_title="Agentic RAG Assistant", page_icon="⚡", layout="wide")

    st.markdown("""
    <style>
        .stApp {
            background-color: #0d1117;
            color: #c9d1d9;
            font-family: 'Inter', sans-serif;
        }
        [data-testid="stSidebar"] {
            background-color: #161b22;
            border-right: 1px solid #30363d;
        }
        .banner {
            padding: 2rem;
            background: linear-gradient(135deg, #1f6feb 0%, #111e38 100%);
            border-radius: 12px;
            border: 1px solid #30363d;
            margin-bottom: 2rem;
            text-align: center;
        }
        .banner h1 {
            color: #ffffff;
            font-size: 2.2rem;
            font-weight: 700;
            margin: 0;
        }
        .banner p {
            color: #8b949e;
            font-size: 1rem;
            margin-top: 0.5rem;
        }
    </style>
    """, unsafe_allow_html=True)

    # Title Banner
    st.markdown("""
    <div class="banner">
        <h1>⚡ Agentic RAG Assistant</h1>
        <p>Hybrid Search (Dense + Sparse) & RRF Fusion powered by Qdrant & LangChain</p>
    </div>
    """, unsafe_allow_html=True)

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
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Sidebar for File Upload
    st.sidebar.markdown("### Document Indexing")
    uploaded_file = st.sidebar.file_uploader("Upload PDF Document", type=["pdf"])

    if uploaded_file:
        os.makedirs("documents", exist_ok=True)
        temp_path = os.path.join("documents", uploaded_file.name)
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Process and store document
        if not st.session_state.agent:
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
            st.sidebar.success("Document processed successfully! Agent is ready.")

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat Interface
    user_input = st.chat_input("Ask a question about the uploaded document...")

    if user_input and st.session_state.agent:
        # Display user message
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)
        
        # Display assistant response with streaming
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            full_response = ""
            
            # Stream the response using agent.stream
            for token in stream_agent_response(
                st.session_state.agent,
                user_input,
                thread_id="session_001"  # Use consistent thread_id for memory
            ):
                full_response += token
                response_placeholder.markdown(full_response + "▌")
            
            # Final update without cursor
            response_placeholder.markdown(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            
    elif user_input and not st.session_state.agent:
        st.warning("Please upload a document first!")

if __name__ == "__main__":
    main()
