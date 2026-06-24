from qdrant_client import QdrantClient, models

QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "market_research_collection"

def setup_qdrant_collection(chunks, dense_embedder, sparse_embedder):
    """Sets up a hybrid Qdrant collection and indexes document chunks."""
    client = QdrantClient(url=QDRANT_URL)
    
    client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={
            "dense": models.VectorParams(
                size=768, 
                distance=models.Distance.COSINE
            )
        },
        sparse_vectors_config={
            "sparse": models.SparseVectorParams()
        }
    )
    
    texts = [chunk.page_content for chunk in chunks]
    dense_embeddings = list(dense_embedder.embed(texts))
    sparse_embeddings = list(sparse_embedder.embed(texts))
    
    points = []
    for idx, chunk in enumerate(chunks):
        sparse_val = sparse_embeddings[idx]
        points.append(
            models.PointStruct(
                id=idx,
                vector={
                    "dense": dense_embeddings[idx].tolist(),
                    "sparse": models.SparseVector(
                        indices=sparse_val.indices.tolist(),
                        values=sparse_val.values.tolist()
                    )
                },
                payload={
                    "page_content": chunk.page_content,
                    "metadata": chunk.metadata
                }
            )
        )
        
    client.upload_points(collection_name=COLLECTION_NAME, points=points)
    return client