from qdrant_client import models


def hybrid_search_rrf(
    client, collection_name, query_text, dense_embedder, sparse_embedder, limit=5
):
    """Perform hybrid search using dense + sparse vectors with RRF fusion.

    Uses the native Qdrant Query API with prefetch and RRF fusion.
    """
    # TODO: Generate embeddings for the query
    dense_query = list(dense_embedder.embed([query_text], batch_size=1))[0]
    sparse_query = list(sparse_embedder.embed([query_text], batch_size=1))[0]

    # Convert sparse vector
    sparse_indices = (
        sparse_query.indices.tolist()
        if hasattr(sparse_query.indices, "tolist")
        else list(sparse_query.indices)
    )
    sparse_values = (
        sparse_query.values.tolist()
        if hasattr(sparse_query.values, "tolist")
        else list(sparse_query.values)
    )
    dense_vector = dense_query.tolist() if hasattr(dense_query, "tolist") else list(dense_query)

    # TODO: Perform hybrid search with RRF fusion using prefetch
    results = client.query_points(
        collection_name=collection_name,
        prefetch=[
            # Sparse vector prefetch (keyword search)
            models.Prefetch(
                query=models.SparseVector(indices=sparse_indices, values=sparse_values),
                using="sparse",
                limit=20,  # Retrieve top 20 from sparse search
            ),
            # Dense vector prefetch (semantic search)
            models.Prefetch(
                query=dense_vector,
                using="dense",
                limit=20,  # Retrieve top 20 from dense search
            ),
        ],
        # TODO: Apply RRF fusion to combine results
        query=models.FusionQuery(fusion=models.Fusion.RRF),  # Reciprocal Rank Fusion
        with_payload=True,
        limit=limit,  # Final number of results to return
    )

    # TODO: Extract and return the documents
    documents = []
    for point in results.points:
        payload = point.payload or {}
        doc = {
            "content": payload.get("page_content", ""),
            "metadata": payload.get("metadata", {}),
            "score": point.score,
            "id": point.id,
        }
        documents.append(doc)

    return documents
