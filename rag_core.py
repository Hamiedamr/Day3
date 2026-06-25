"""Pure RAG logic shared by the MCP server and the ingestion script.

No agent, no MCP, no UI — just document loading, chunking, embedding,
and native Qdrant hybrid search.
"""
import uuid

from langchain_pymupdf4llm import PyMuPDF4LLMLoader
from langchain_text_splitters import MarkdownTextSplitter
from qdrant_client import QdrantClient, models
from fastembed import TextEmbedding, SparseTextEmbedding

QDRANT_URL = "http://localhost:6333"
DENSE_MODEL = "jinaai/jina-embeddings-v2-base-en"
SPARSE_MODEL = "Qdrant/bm25"


def load_and_split_pdf(file_path):
    """Load a PDF and split it into markdown chunks."""
    loader = PyMuPDF4LLMLoader(file_path=file_path)
    docs = loader.load()
    splitter = MarkdownTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(docs)
    return chunks


def setup_qdrant_collection(chunks, collection_name="documents"):
    """Create the Qdrant collection and upload dense + sparse vectors."""
    client = QdrantClient(url=QDRANT_URL)

    dense_embedder = TextEmbedding(model_name=DENSE_MODEL)
    sparse_embedder = SparseTextEmbedding(model_name=SPARSE_MODEL)

    sample_embedding = list(dense_embedder.embed(["sample text"]))[0]
    vector_size = len(sample_embedding)

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

    client.upload_points(
        collection_name=collection_name,
        points=points,
        batch_size=64,
        parallel=2,
        max_retries=3,
        wait=False
    )

    return collection_name


def hybrid_search_rrf(query_text, collection_name="documents", limit=5,
                      client=None, dense_embedder=None, sparse_embedder=None):
    """Hybrid search using dense + sparse vectors with RRF fusion.

    Connections/embedders are built lazily so the MCP tool can call this
    statelessly.
    """
    client = client or QdrantClient(url=QDRANT_URL)
    dense_embedder = dense_embedder or TextEmbedding(model_name=DENSE_MODEL)
    sparse_embedder = sparse_embedder or SparseTextEmbedding(model_name=SPARSE_MODEL)

    dense_query = list(dense_embedder.embed([query_text]))[0]
    sparse_query = list(sparse_embedder.embed([query_text]))[0]

    sparse_indices = sparse_query.indices.tolist()
    sparse_values = sparse_query.values.tolist()

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

    documents = []
    for point in results.points:
        documents.append({
            "content": point.payload["page_content"],
            "metadata": point.payload["metadata"],
            "score": point.score,
            "id": point.id
        })

    return documents
