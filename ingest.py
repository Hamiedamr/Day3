"""Offline document ingestion stage.

Loads PDFs, splits into chunks, embeds with dense+sparse vectors,
and uploads to the native Qdrant collection.

Usage:  uv run python ingest.py <path_to_pdf>
"""
import sys
import uuid

from langchain_pymupdf4llm import PyMuPDF4LLMLoader
from langchain_text_splitters import MarkdownTextSplitter
from qdrant_client import QdrantClient, models
from fastembed import TextEmbedding, SparseTextEmbedding

COLLECTION_NAME = "my_agentic_rag_collection"
DENSE_MODEL = "jinaai/jina-embeddings-v2-base-en"
SPARSE_MODEL = "Qdrant/bm25"
QDRANT_URL = "http://localhost:6333"

# Module-level state — populated by setup_qdrant_collection, consumed by app.py
_search_client = None
_search_collection = None
_search_dense = None
_search_sparse = None


def load_and_split_pdf(file_path):
    loader = PyMuPDF4LLMLoader(file_path=file_path)
    docs = loader.load()
    splitter = MarkdownTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(docs)
    return chunks


def setup_qdrant_collection(chunks, collection_name=COLLECTION_NAME):
    """Setup Qdrant collection using native client with dense + sparse vectors.

    Returns (client, collection_name, dense_embedder, sparse_embedder)
    so the online stage can reuse the embedders for search.
    """
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

    global _search_client, _search_collection, _search_dense, _search_sparse
    _search_client = client
    _search_collection = collection_name
    _search_dense = dense_embedder
    _search_sparse = sparse_embedder

    return client, collection_name, dense_embedder, sparse_embedder


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: uv run python ingest.py <path_to_pdf>")
        sys.exit(1)

    pdf_path = sys.argv[1]
    print(f"Ingesting: {pdf_path}")

    chunks = load_and_split_pdf(pdf_path)
    print(f"Split into {len(chunks)} chunks")

    client, collection_name, _, _ = setup_qdrant_collection(chunks)
    count = client.count(collection_name).count
    print(f"Uploaded {count} points to collection '{collection_name}'")
    print("Ingestion complete.")
