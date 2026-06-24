from fastembed import SparseTextEmbedding, TextEmbedding
from qdrant_client import QdrantClient, models

from rag.config import (
    COLLECTION_NAME,
    DENSE_EMBED_MODEL,
    QDRANT_URL,
    SPARSE_EMBED_MODEL,
)


def hybrid_search_rrf(
    query_text,
    collection_name=COLLECTION_NAME,
    limit=5,
    client=None,
    dense_embedder=None,
    sparse_embedder=None,
):
    client = client or QdrantClient(url=QDRANT_URL)
    dense_embedder = dense_embedder or TextEmbedding(model_name=DENSE_EMBED_MODEL)
    sparse_embedder = sparse_embedder or SparseTextEmbedding(model_name=SPARSE_EMBED_MODEL)

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
                    values=sparse_values,
                ),
                using="sparse",
                limit=20,
            ),
            models.Prefetch(
                query=dense_query.tolist() if hasattr(dense_query, "tolist") else list(dense_query),
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
        doc = {
            "content": point.payload["page_content"],
            "metadata": point.payload["metadata"],
            "score": point.score,
            "id": point.id,
        }
        documents.append(doc)

    return documents
