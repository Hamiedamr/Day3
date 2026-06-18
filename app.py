from langchain_pymupdf4llm import PyMuPDF4LLMLoader
from langchain_text_splitters import MarkdownTextSplitter

def load_and_split_pdf(file_path):
    # TODO: Initialize PyMuPDF4LLMLoader with the given file path
    loader = PyMuPDF4LLMLoader(file_path=file_path)
    
    # TODO: Load the document
    docs = loader.load()
    
    # TODO: Initialize MarkdownTextSplitter with appropriate chunk settings
    splitter = MarkdownTextSplitter(chunk_size=1000, chunk_overlap=200)
    
    # TODO: Split the documents into chunks
    chunks = splitter.split_documents(docs)
    
    return chunks

#_________________________Task8_____________________________________________

from qdrant_client import QdrantClient, models
from fastembed import TextEmbedding, SparseTextEmbedding
import uuid

def setup_qdrant_collection(chunks):
    """Setup Qdrant collection using native client with dense + sparse vectors.
    
    Uses FastEmbed for generating embeddings locally.
    """
    # TODO: Initialize Native Qdrant Client
    client = QdrantClient(url="http://localhost:6333")
    
    # TODO: Initialize embedding models (FastEmbed)
    dense_embedding_model = "jinaai/jina-embeddings-v2-base-en"  # e.g., "jinaai/jina-embeddings-v2-base-en"
    sparse_embedding_model = "Qdrant/bm25"  # e.g., "Qdrant/bm25" or "prithivida/Splade_PP_en_v1"
    
    dense_embedder = TextEmbedding(model_name=dense_embedding_model)
    sparse_embedder = SparseTextEmbedding(model_name=sparse_embedding_model)
    
    # TODO: Create collection with explicit vector configurations
    collection_name = "my_documents"
    
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
    
    # TODO: Prepare points for upload with dense + sparse vectors
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
    
    # TODO: Upload points using native upload_points (with parallelization)
    client.upload_points(
        collection_name=collection_name,
        points=points,
        batch_size=64,      # e.g., 64 (default)
        parallel=2,        # e.g., 2 workers
        max_retries=3,     # e.g., 3
        wait=True           # Async mode for better performance
    )
    
    return client, collection_name, dense_embedder, sparse_embedder

#____________________________________Task9____________________________
def hybrid_search_rrf(client, collection_name, query_text, dense_embedder, sparse_embedder, limit=5):
    """Perform hybrid search using dense + sparse vectors with RRF fusion.
    
    Uses the native Qdrant Query API with prefetch and RRF fusion.
    """
    # TODO: Generate embeddings for the query
    dense_query = list(dense_embedder.embed([query_text]))[0]
    sparse_query = list(sparse_embedder.embed([query_text]))[0]
    
    # Convert sparse vector
    sparse_indices = sparse_query.indices.tolist()
    sparse_values = sparse_query.values.tolist()
    
    # TODO: Perform hybrid search with RRF fusion using prefetch
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
        # TODO: Apply RRF fusion to combine results
        query=models.FusionQuery(
            fusion=models.Fusion.RRF  # Reciprocal Rank Fusion
        ),
        with_payload=True,
        limit=limit  # Final number of results to return
    )
    
    # TODO: Extract and return the documents
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

#________________________________________Task10__________________________________________________________
from langchain.agents import create_agent  # 2026 syntax
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver  # For conversation memory

def create_rag_agent(qdrant_client, collection_name, dense_embedder, sparse_embedder):
    """Create an agentic RAG agent with native Qdrant hybrid search tool."""
    
    # TODO: Define the retrieval tool using native Qdrant hybrid search
    @tool(response_format="content_and_artifact")  # Use "content_and_artifact" for best results
    def retrieve_context(query: str):
        """Retrieve relevant documents using hybrid search with RRF fusion.
        
        Args:
            query: The search query to find relevant documents
        """
        # TODO: Call the hybrid_search_rrf function
        docs = hybrid_search_rrf(
            client=qdrant_client,
            collection_name=collection_name,
            query_text=query,
            dense_embedder=dense_embedder,
            sparse_embedder=sparse_embedder,
            limit=5
        )
        
        # TODO: Format the retrieved documents
        serialized = "\n\n".join(
            f"Source: {doc['metadata']}\nContent: {doc['content']}\nScore: {doc['score']}"
            for doc in docs
        )
        return serialized, docs
    
    # TODO: Define the system prompt
    system_prompt = """
    You are a helpful AI assistant with access to a document knowledge base.
    
    Instructions:
    - Use the retrieve_context tool when you need information from the documents
    - The retrieval uses hybrid search (semantic + keyword) with RRF fusion for best results
    - Always cite your sources when using retrieved information
    - If the retrieved context doesn't contain relevant information, say "I don't have enough information to answer that question"
    - You can ask follow-up questions if the query is unclear
    """
    
    # TODO: Create the agent using create_agent (2026 syntax)
    # The model string "ollama:<model>" is used directly — no separate ChatOllama import needed
    agent = create_agent(
        model="ollama:qwen3:4b-instruct",           # e.g., "ollama:qwen3:4b-instruct" (provider:model string)
        tools=[retrieve_context],           # List containing the retrieve_context tool
        system_prompt=system_prompt,   # System instructions
        checkpointer=InMemorySaver(),    # InMemorySaver() for conversation persistence
    )
    
    return agent

#__________________________________________Task11_______________________________________________________________
def stream_agent_response(agent, user_query, thread_id="default"):
    """Stream the agent response with stream_mode='values' (2026 recommended approach).
    
    Args:
        agent: The created agent
        user_query: The user's question
        thread_id: Conversation thread ID for persistence
    """
    from langchain.messages import AIMessage, HumanMessage
    
    # TODO: Prepare the input messages (2026 format)
    inputs = {
        "messages": [{"role": "user", "content": user_query}]
    }
    
    # TODO: Prepare config with thread_id for conversation memory
    config = {
        "configurable": {
            "thread_id": thread_id  # Required for checkpointer to work
        }
    }
    
    # TODO: Stream with stream_mode="values" (2026 recommended approach)
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


#__________________________________________Task12_______________________________________________________________
# Task 12: Build the Streamlit UI with Streaming
# Your Task: Build the complete Streamlit interface:

import streamlit as st
import os
from qdrant_client import QdrantClient

# TODO: Import your functions

# TODO: Create page title
st.title("🤖 Agentic RAG Chatbot (Hybrid Search)")

# TODO: Initialize session state for agent and Qdrant components
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

# TODO: Create Sidebar for File Upload
uploaded_file = st.sidebar.file_uploader("Upload your PDF document", type=["pdf"])

if uploaded_file:
    # TODO: Save file locally
    os.makedirs("documents", exist_ok=True)
    temp_path = os.path.join("documents", uploaded_file.name)
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # TODO: Process and store document
    if st.session_state.agent is None:
        with st.spinner("Processing document with native Qdrant..."):
            # Load and split PDF
            chunks = load_and_split_pdf(temp_path)
            
            # Setup native Qdrant collection with dense + sparse vectors
            qdrant_client, collection_name, dense_embedder, sparse_embedder = setup_qdrant_collection(chunks)
            
            st.session_state.qdrant_client = qdrant_client
            st.session_state.collection_name = collection_name
            st.session_state.dense_embedder = dense_embedder
            st.session_state.sparse_embedder = sparse_embedder
            
            # TODO: Create the agent with native Qdrant components
            st.session_state.agent = create_rag_agent(
                qdrant_client,
                collection_name,
                dense_embedder,
                sparse_embedder
            )
        st.success("Document processed with hybrid search (RRF fusion)! Agent is ready.")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# TODO: Create Chat Interface
user_input = st.chat_input("Ask me anything about the document...")

if user_input and st.session_state.agent:
    # TODO: Display user message
    st.chat_message("user").write(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # TODO: Display assistant response with streaming
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""
        
        # TODO: Stream the response using agent.stream (synchronous — no async needed)
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