"""Shared RAG logic for the MCP-based Agentic RAG system.

This module holds the pure RAG logic (no agent, no MCP) so both the MCP server
and the ingestion step can reuse it.

- Document loading & text splitting
- Native Qdrant setup — collection creation & points insertion
- Native hybrid search with RRF fusion
"""

from langchain_pymupdf4llm import PyMuPDF4LLMLoader
from langchain_text_splitters import MarkdownTextSplitter
from qdrant_client import QdrantClient, models
from fastembed import TextEmbedding, SparseTextEmbedding
import uuid

QDRANT_URL = "http://localhost:6333"
DENSE_MODEL = "jinaai/jina-embeddings-v2-base-en"
SPARSE_MODEL = "Qdrant/bm25"


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


def setup_qdrant_collection(chunks, collection_name="documents"):
    """Setup Qdrant collection using native client with dense + sparse vectors.

    Returns only the collection name — embedders are not returned because the
    MCP server recreates them from DENSE_MODEL / SPARSE_MODEL on its own.
    """
    # Initialize Native Qdrant Client
    client = QdrantClient(url=QDRANT_URL)

    # Initialize embedding models (FastEmbed)
    dense_embedder = TextEmbedding(model_name=DENSE_MODEL)
    sparse_embedder = SparseTextEmbedding(model_name=SPARSE_MODEL)

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

    dense_vectors = list(dense_embedder.embed(texts))
    sparse_vectors = list(sparse_embedder.embed(texts))

    for idx, (chunk, dense_vec, sparse_vec) in enumerate(zip(chunks, dense_vectors, sparse_vectors)):
        sparse_indices = sparse_vec.indices.tolist()
        sparse_values = sparse_vec.values.tolist()

        point = models.PointStruct(
            id=str(uuid.uuid4()),
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
        batch_size=64,      # e.g., 64
        parallel=2,         # e.g., 2 workers
        max_retries=3,      # e.g., 3
        wait=False
    )

    return collection_name


def hybrid_search_rrf(query_text, collection_name="documents", limit=5,
                      client=None, dense_embedder=None, sparse_embedder=None):
    """Perform hybrid search using dense + sparse vectors with RRF fusion."""
    # Lazily build connections/embedders so the MCP tool can call this statelessly
    client = client or QdrantClient(url=QDRANT_URL)
    dense_embedder = dense_embedder or TextEmbedding(model_name=DENSE_MODEL)
    sparse_embedder = sparse_embedder or SparseTextEmbedding(model_name=SPARSE_MODEL)

    # Generate embeddings for the query
    dense_query = list(dense_embedder.embed([query_text]))[0]
    sparse_query = list(sparse_embedder.embed([query_text]))[0]

    sparse_indices = sparse_query.indices.tolist()
    sparse_values = sparse_query.values.tolist()

    # Perform hybrid search with RRF fusion using prefetch
    results = client.query_points(
        collection_name=collection_name,
        prefetch=[
            models.Prefetch(
                query=models.SparseVector(indices=sparse_indices, values=sparse_values),
                using="sparse",
                limit=20
            ),
            models.Prefetch(
                query=dense_query.tolist() if hasattr(dense_query, 'tolist') else list(dense_query),
                using="dense",
                limit=20
            )
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        with_payload=True,
        limit=limit
    )

    # Extract and return the documents
    documents = []
    for point in results.points:
        documents.append({
            "content": point.payload["page_content"],
            "metadata": point.payload["metadata"],
            "score": point.score,
            "id": point.id
        })

    return documents
