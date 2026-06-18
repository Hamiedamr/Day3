import uuid
from qdrant_client import QdrantClient, models
from fastembed import TextEmbedding, SparseTextEmbedding

def hybrid_search_rrf(query_text, limit=3):
    """
    Perform hybrid search using dense + sparse vectors with RRF fusion.
    Uses the native Qdrant Query API with prefetch and RRF fusion.
    """
    client = QdrantClient(url="http://localhost:6333")
    collection_name = "nesma_cv_collection"
    
    # Matching the exact local model used in app.py
    dense_embedder = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
    sparse_embedder = SparseTextEmbedding(model_name="Qdrant/bm25")
    
    # 1. Generate embeddings for the query
    dense_query = list(dense_embedder.embed([query_text]))[0]
    sparse_query = list(sparse_embedder.embed([query_text]))[0]
    
    # Convert sparse vector indices and values
    sparse_indices = sparse_query.indices.tolist()
    sparse_values = sparse_query.values.tolist()
    
    # 2. Perform hybrid search with RRF fusion using prefetch
    results = client.query_points(
        collection_name=collection_name,
        prefetch=[
            # Sparse vector prefetch (keyword search)
            models.Prefetch(
                query=models.SparseVector(
                    indices=sparse_indices,
                    values=sparse_values
                ),
                using="sparse",
                limit=20  
            ),
            # Dense vector prefetch (semantic search)
            models.Prefetch(
                query=dense_query.tolist() if hasattr(dense_query, 'tolist') else list(dense_query),
                using="dense",
                limit=20  
            )
        ],
        # 3. Apply RRF fusion to combine results
        query=models.FusionQuery(
            fusion=models.Fusion.RRF  
        ),
        with_payload=True,
        limit=limit  
    )
    
    # 4. Extract and return the documents
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
    test_query = "What are the technical skills and frameworks used?"
    
    print(f"Searching for: '{test_query}'...\n")
    search_results = hybrid_search_rrf(query_text=test_query, limit=2)
    
    print(f"--- Found {len(search_results)} relevant chunks ---\n")
    for idx, doc in enumerate(search_results):
        print(f"Result {idx + 1} (RRF Score: {doc['score']:.4f}):")
        print(doc['content'])
        print("-" * 50)