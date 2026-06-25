from qdrant_client import QdrantClient, models
from fastembed import TextEmbedding, SparseTextEmbedding

def hybrid_search_rrf(client, collection_name, query_text, dense_embedder, sparse_embedder, limit=5):
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
        doc = {
            "content": point.payload["page_content"],
            "metadata": point.payload["metadata"],
            "score": point.score,
            "id": point.id
        }
        documents.append(doc)
    return documents

def search_documents(query_text, limit=5):
    client = QdrantClient(url="http://localhost:6333", check_compatibility=False)
    dense_embedder = TextEmbedding(model_name="jinaai/jina-embeddings-v2-base-en")
    sparse_embedder = SparseTextEmbedding(model_name="Qdrant/bm25")
    return hybrid_search_rrf(client, "agentic_rag_collection", query_text, dense_embedder, sparse_embedder, limit)

if __name__ == "__main__":
    import sys
    query = sys.argv[1] if len(sys.argv) > 1 else "What is asthma and how is it predicted?"
    results = search_documents(query, limit=5)
    for i, doc in enumerate(results):
        print(f"\n--- Result {i+1} (score: {doc['score']:.4f}) ---")
        print(doc["content"][:300])
