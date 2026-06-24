from qdrant_client import QdrantClient, models

QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "market_reseachr_collection"

def search_vault(query_text: str, dense_embedder, sparse_embedder, limit: int = 3):
    """Retrieves document nodes matching user query via Reciprocal Rank Fusion (RRF)."""
    client = QdrantClient(url=QDRANT_URL)
    
    dense_query = list(dense_embedder.embed([query_text]))[0]
    sparse_query = list(sparse_embedder.embed([query_text]))[0]
    
    results = client.query_points(
        collection_name=COLLECTION_NAME,
        prefetch=[
            models.Prefetch(
                query=models.SparseVector(
                    indices=sparse_query.indices.tolist(),
                    values=sparse_query.values.tolist()
                ),
                using="sparse",
                limit=20
            ),
            models.Prefetch(
                query=dense_query.tolist(),
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
            "score": point.score
        })
    return documents