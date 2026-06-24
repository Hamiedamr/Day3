from qdrant_client import models

def hybrid_search_rrf(client, collection_name, query_text, dense_embedder, sparse_embedder, limit=5):
    """
    Performs a native hybrid search using Dense (Semantic) and Sparse (Keyword) 
    vectors, combining their rankings using Reciprocal Rank Fusion (RRF).
    """
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
                    values=sparse_values
                ),
                using="sparse",
                limit=20  
            ),
            models.Prefetch(
                query=dense_query.tolist() if hasattr(dense_query, 'tolist') else list(dense_query),
                using="dense",
                limit=20  
            )
        ],
        query=models.FusionQuery(
            fusion=models.Fusion.RRF  
        ),
        with_payload=True,
        limit=limit  
    )
    
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