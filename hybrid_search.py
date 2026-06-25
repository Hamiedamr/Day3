from qdrant_client import models


def hybrid_search_rrf(client, collection_name, query_text, dense_embedder, sparse_embedder, limit=5):
    """Perform hybrid search using dense + sparse vectors with RRF fusion.

    Uses the native Qdrant Query API with prefetch and RRF fusion.
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


if __name__ == "__main__":
    import time
    from load_split import load_and_split_pdf
    from qdrant_setup import setup_qdrant_collection

    chunks = load_and_split_pdf("documents/Mahmoud_Ramadan_Abbas_CV.pdf")
    client, collection_name, dense_embedder, sparse_embedder = setup_qdrant_collection(chunks)

    time.sleep(2) 

    results = hybrid_search_rrf(
        client=client,
        collection_name=collection_name,
        query_text="What is this document about?",
        dense_embedder=dense_embedder,
        sparse_embedder=sparse_embedder,
        limit=3
    )

    for i, doc in enumerate(results):
        print(f"\n--- Result {i+1} (score: {doc['score']:.4f}) ---")
        print(doc['content'][:200])
        print(doc['metadata'])