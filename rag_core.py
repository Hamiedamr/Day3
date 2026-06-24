import uuid
from langchain_pymupdf4llm import PyMuPDF4LLMLoader
from langchain_text_splitters import MarkdownTextSplitter
from qdrant_client import QdrantClient, models
from fastembed import TextEmbedding, SparseTextEmbedding
from langchain.agents import create_agent
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langchain.messages import AIMessage


def load_and_split_pdf(file_path, chunk_size=1000, chunk_overlap=200):
    loader = PyMuPDF4LLMLoader(file_path=file_path)
    docs = loader.load()
    splitter = MarkdownTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(docs)


def setup_qdrant_collection(chunks, collection_name="demo_rag", dense_model="BAAI/bge-small-en-v1.5", sparse_model="Qdrant/bm25"):
    client = QdrantClient(url="http://localhost:6333", timeout=60)
    dense_embedder = TextEmbedding(model_name=dense_model)
    sparse_embedder = SparseTextEmbedding(model_name=sparse_model)

    sample_embedding = list(dense_embedder.embed(["sample text"]))[0]
    vector_size = len(sample_embedding)

    if not client.collection_exists(collection_name=collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config={
                "dense": models.VectorParams(size=vector_size, distance=models.Distance.COSINE)
            },
            sparse_vectors_config={
                "sparse": models.SparseVectorParams(index=models.SparseIndexParams(on_disk=False))
            }
        )

    collection_info = client.get_collection(collection_name)
    if collection_info.points_count == 0:
        points = []
        texts = [chunk.page_content for chunk in chunks]
        dense_vectors = list(dense_embedder.embed(texts))
        sparse_vectors = list(sparse_embedder.embed(texts))

        for idx, (chunk, dense_vec, sparse_vec) in enumerate(zip(chunks, dense_vectors, sparse_vectors)):
            sparse_indices = sparse_vec.indices.tolist()
            sparse_values = sparse_vec.values.tolist()
            point = models.PointStruct(
                id=str(uuid.uuid4()),
                vector={
                    "dense": dense_vec.tolist() if hasattr(dense_vec, 'tolist') else list(dense_vec),
                    "sparse": models.SparseVector(indices=sparse_indices, values=sparse_values)
                },
                payload={"page_content": chunk.page_content, "metadata": chunk.metadata, "chunk_id": idx}
            )
            points.append(point)

        client.upload_points(
            collection_name=collection_name, points=points,
            batch_size=64, parallel=2, max_retries=3, wait=True
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
                query=models.SparseVector(indices=sparse_indices, values=sparse_values),
                using="sparse", limit=20
            ),
            models.Prefetch(
                query=dense_query.tolist() if hasattr(dense_query, 'tolist') else list(dense_query),
                using="dense", limit=20
            )
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        with_payload=True, limit=limit
    )

    return [
        {
            "content": point.payload["page_content"],
            "metadata": point.payload["metadata"],
            "score": point.score,
            "id": point.id
        }
        for point in results.points
    ]


def create_rag_agent(client=None, collection_name=None, dense_embedder=None, sparse_embedder=None, checkpointer=None, system_prompt=None):
    if client is None:
        client = QdrantClient(url="http://localhost:6333", timeout=60)
    if collection_name is None:
        collection_name = "demo_rag"
    if dense_embedder is None:
        dense_embedder = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
    if sparse_embedder is None:
        sparse_embedder = SparseTextEmbedding(model_name="Qdrant/bm25")
    if system_prompt is None:
        system_prompt = (
            "You are a specialized search agent.\n"
            "CRITICAL: You do not know anything about React by default. "
            "You MUST use the retrieve_context tool to search for information "
            "before you can answer ANY user question. Do not answer from your own knowledge. "
            "Always query the retrieve_context tool first."
        )

    @tool(response_format="content_and_artifact")
    def retrieve_context(query: str) -> tuple[str, list]:
        """Retrieve relevant documents from the knowledge base using hybrid search with RRF fusion.

        Args:
            query: The search query to find relevant documents.
        """
        docs = hybrid_search_rrf(client, collection_name, query, dense_embedder, sparse_embedder, limit=5)
        serialized = "\n\n".join(
            f"Source: {doc['metadata']}\nContent: {doc['content']}\nScore: {doc['score']}"
            for doc in docs
        )
        return serialized, docs

    agent = create_agent(
        model="ollama:qwen3:4b-instruct",
        tools=[retrieve_context],
        system_prompt=system_prompt,
        checkpointer=checkpointer if checkpointer is not None else InMemorySaver(),
    )
    return agent


def stream_agent_response(agent, user_query, thread_id="default"):
    inputs = {"messages": [{"role": "user", "content": user_query}]}
    config = {"configurable": {"thread_id": thread_id}}

    for chunk in agent.stream(inputs, stream_mode="values", config=config):
        latest_message = chunk["messages"][-1]
        if isinstance(latest_message, AIMessage) and latest_message.content:
            yield latest_message.content
        elif hasattr(latest_message, 'tool_calls') and latest_message.tool_calls:
            yield f"\nSearching: {[tc['name'] for tc in latest_message.tool_calls]}\n"
