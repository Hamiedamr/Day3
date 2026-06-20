from qdrant_client import models

def hybrid_search_rrf(client, collection_name, query_text, dense_embedder, sparse_embedder, limit=5):
    # generate dense and sparse embeddings
    dense_vec = list(dense_embedder.embed([query_text]))[0]
    sparse_vec = list(sparse_embedder.embed([query_text]))[0]
    
    results = client.query_points(
        collection_name=collection_name,
        prefetch=[
            models.Prefetch(
                query=models.SparseVector(
                    indices=sparse_vec.indices.tolist(),
                    values=sparse_vec.values.tolist()
                ),
                using="sparse",
                limit=20
            ),
            models.Prefetch(
                query=dense_vec.tolist() if hasattr(dense_vec, 'tolist') else list(dense_vec),
                using="dense",
                limit=20
            )
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        with_payload=True,
        limit=limit
    )
    
    # extract and return docs
    docs = []
    for point in results.points:
        docs.append({
            "content": point.payload["page_content"],
            "metadata": point.payload["metadata"],
            "score": point.score,
            "id": point.id
        })
    
    return docs
