from qdrant_client import QdrantClient, models
from fastembed import TextEmbedding, SparseTextEmbedding
import uuid

BATCH_SIZE = 4


def setup_qdrant_collection(chunks):
    """Setup Qdrant collection using native client with dense + sparse vectors.

    Uses FastEmbed for generating embeddings locally.
    """
    # TODO: Initialize Native Qdrant Client
    client = QdrantClient(url="http://localhost:6333")

    # TODO: Initialize embedding models (FastEmbed)
    dense_embedding_model = "jinaai/jina-embeddings-v2-base-en"
    sparse_embedding_model = "Qdrant/bm25"

    dense_embedder = TextEmbedding(model_name=dense_embedding_model)
    sparse_embedder = SparseTextEmbedding(model_name=sparse_embedding_model)

    # TODO: Create collection with explicit vector configurations
    collection_name = "agentic_rag_chunks2"

    # Get embedding dimensions by encoding a sample text
    sample_embedding = list(dense_embedder.embed(["sample text"]))[0]
    vector_size = len(sample_embedding)

    # Create collection if it doesn't exist
    if not client.collection_exists(collection_name=collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config={
                "dense": models.VectorParams(
                    size=vector_size, distance=models.Distance.COSINE
                )
            },
            sparse_vectors_config={
                "sparse": models.SparseVectorParams(
                    index=models.SparseIndexParams(on_disk=False)
                )
            },
        )

    # Embed and upload in small batches to avoid ONNXRuntime memory spikes.
    for batch_start in range(0, len(chunks), BATCH_SIZE):
        batch_chunks = chunks[batch_start : batch_start + BATCH_SIZE]
        texts = [chunk.page_content for chunk in batch_chunks]
        dense_vectors = list(dense_embedder.embed(texts, batch_size=BATCH_SIZE))
        sparse_vectors = list(sparse_embedder.embed(texts, batch_size=BATCH_SIZE))
        points = []

        for idx, (chunk, dense_vec, sparse_vec) in enumerate(
            zip(batch_chunks, dense_vectors, sparse_vectors), start=batch_start
        ):
            # FastEmbed sparse vectors expose NumPy arrays for indices and values.
            sparse_indices = sparse_vec.indices.tolist()
            sparse_values = sparse_vec.values.tolist()

            point = models.PointStruct(
                id=str(uuid.uuid4()),  # or use idx for integer IDs
                vector={
                    "dense": (
                        dense_vec.tolist()
                        if hasattr(dense_vec, "tolist")
                        else list(dense_vec)
                    ),
                    "sparse": models.SparseVector(
                        indices=sparse_indices, values=sparse_values
                    ),
                },
                payload={
                    "page_content": chunk.page_content,
                    "metadata": chunk.metadata,
                    "chunk_id": idx,
                },
            )
            points.append(point)

        # TODO: Upload points using native upload_points (with parallelization)
        client.upload_points(
            collection_name=collection_name,
            points=points,
            batch_size=BATCH_SIZE,
            parallel=1,
            max_retries=3,  # e.g., 3
            wait=True,
        )

    return client, collection_name, dense_embedder, sparse_embedder
