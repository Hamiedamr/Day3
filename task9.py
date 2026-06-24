from qdrant_client import QdrantClient, models
from fastembed import TextEmbedding, SparseTextEmbedding

def hybrid_search_rrf(query_text, limit=5):
    """Perform hybrid search using dense + sparse vectors with RRF fusion.
    Uses the native Qdrant Query API with prefetch and RRF fusion.
    """
    client          = QdrantClient(url="http://localhost:6333", timeout=60)
    collection_name = "rag_hybrid_collection"
    
    dense_embedder  = TextEmbedding(model_name="jinaai/jina-embeddings-v2-base-en")
    sparse_embedder = SparseTextEmbedding(model_name="Qdrant/bm25")

    dense_query  = list(dense_embedder.embed([query_text]))[0]
    sparse_query = list(sparse_embedder.embed([query_text]))[0]
    
    sparse_indices = sparse_query.indices.tolist()
    sparse_values  = sparse_query.values.tolist()
    
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
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        with_payload=True,
        limit=limit
    )
    
    documents = []
    for point in results.points:
        doc = {
            "content":  point.payload["page_content"],
            "metadata": point.payload["metadata"],
            "score":    point.score,
            "id":       point.id
        }
        documents.append(doc)
    
    return documents

if __name__ == "__main__":
    try:
        search_results = hybrid_search_rrf("What is SELINUX?", limit=3)
        print(f"Successfully retrieved {len(search_results)} matching documents via RRF fusion.")
        for idx, res in enumerate(search_results):
            print(f"\n[Result {idx+1}] Score: {res['score']:.4f}")
            print(f"Content: {res['content'][:200]}...")
    except Exception as e:
        print(f"Error executing search: {e}")