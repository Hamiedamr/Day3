import sys
import os
from qdrant_client import QdrantClient, models
from fastembed import TextEmbedding, SparseTextEmbedding

def hybrid_search_rrf(query_text, limit=5):
    """Perform hybrid search using dense + sparse vectors with RRF fusion.
    
    Uses the native Qdrant Query API with prefetch and RRF fusion.
    """
    # Initialize components as variables inside the function
    client = QdrantClient(url="http://localhost:6333")
    collection_name = "javascript_guide"
    
    dense_embedding_model = "jinaai/jina-embeddings-v2-base-en"
    sparse_embedding_model = "Qdrant/bm25"
    
    dense_embedder = TextEmbedding(model_name=dense_embedding_model)
    sparse_embedder = SparseTextEmbedding(model_name=sparse_embedding_model)
    
    # Generate embeddings for the query
    dense_query = list(dense_embedder.embed([query_text]))[0]
    sparse_query = list(sparse_embedder.embed([query_text]))[0]
    
    # Convert sparse vector
    sparse_indices = sparse_query.indices.tolist()
    sparse_values = sparse_query.values.tolist()
    
    # Perform hybrid search with RRF fusion using prefetch
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
                limit=20  # Retrieve top 20 from sparse search
            ),
            # Dense vector prefetch (semantic search)
            models.Prefetch(
                query=dense_query.tolist() if hasattr(dense_query, 'tolist') else list(dense_query),
                using="dense",
                limit=20  # Retrieve top 20 
            )
        ],
        # Apply RRF fusion to combine results
        query=models.FusionQuery(
            fusion=models.Fusion.RRF  # Reciprocal Rank Fusion
        ),
        with_payload=True,
        limit=limit  # Final number of results to return
    )
    
    # Extract and return the documents
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
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    
    query = sys.argv[1] if len(sys.argv) > 1 else "What is JavaScript?"
    print(f"Executing hybrid search for query: '{query}'...")
    
    try:
        results = hybrid_search_rrf(query, limit=3)
        print(f"\nSuccessfully retrieved {len(results)} results:")
        for idx, doc in enumerate(results):
            print(f"\n--- Result {idx + 1} (Score: {doc['score']:.4f}) ---")
            print(f"Source: {doc['metadata'].get('source', 'N/A')}, Page: {doc['metadata'].get('page', 'N/A')}")
            print("-" * 50)
            print(doc['content'][:400] + ("\n..." if len(doc['content']) > 400 else ""))
            print("-" * 50)
            
    except Exception as e:
        print(f"Error executing hybrid search: {e}")
