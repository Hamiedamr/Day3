import uuid
from qdrant_client import QdrantClient, models
from fastembed import TextEmbedding, SparseTextEmbedding

def setup_qdrant_collection(chunks):
    """
    Connect to Qdrant, create a hybrid collection (dense + sparse), and
    upload all chunk embeddings using the native client.
    """
    # Connect to the locally-running Qdrant instance
    client = QdrantClient(url="http://localhost:6333")
 
    # ── Embedding models ───────────────────────────────────────────────────────
    dense_embedding_model  = "jinaai/jina-embeddings-v2-base-en"   # 768-dim dense
    sparse_embedding_model = "Qdrant/bm25"                          # BM25 sparse
 
    dense_embedder  = TextEmbedding(model_name=dense_embedding_model)
    sparse_embedder = SparseTextEmbedding(model_name=sparse_embedding_model)
 
    # ── Collection configuration ───────────────────────────────────────────────
    collection_name = "agentic_rag_docs"
 
    # Determine vector size from a sample embedding
    sample_embedding = list(dense_embedder.embed(["sample text"]))[0]
    vector_size = len(sample_embedding)
 
    # Create collection only if it doesn't exist yet
    if not client.collection_exists(collection_name=collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config={
                "dense": models.VectorParams(
                    size=vector_size,
                    distance=models.Distance.COSINE
                )
            },
            sparse_vectors_config={
                "sparse": models.SparseVectorParams(
                    index=models.SparseIndexParams(on_disk=False)
                )
            }
        )
 
    # ── Prepare & upload points ────────────────────────────────────────────────
    texts = [chunk.page_content for chunk in chunks]
 
    # Generate all embeddings in one pass (batch-efficient)
    dense_vectors  = list(dense_embedder.embed(texts))
    sparse_vectors = list(sparse_embedder.embed(texts))
 
    points = []
    for idx, (chunk, dense_vec, sparse_vec) in enumerate(
            zip(chunks, dense_vectors, sparse_vectors)):
 
        sparse_indices = sparse_vec.indices.tolist()
        sparse_values  = sparse_vec.values.tolist()
 
        point = models.PointStruct(
            id=str(uuid.uuid4()),
            vector={
                "dense": (dense_vec.tolist()
                          if hasattr(dense_vec, "tolist") else list(dense_vec)),
                "sparse": models.SparseVector(
                    indices=sparse_indices,
                    values=sparse_values
                )
            },
            payload={
                "page_content": chunk.page_content,
                "metadata":     chunk.metadata,
                "chunk_id":     idx
            }
        )
        points.append(point)
 
    # Upload with parallelization for speed
    client.upload_points(
        collection_name=collection_name,
        points=points,
        batch_size=64,    
        parallel=2,       
        max_retries=3,    
        wait=False          
    )
 
    return client, collection_name, dense_embedder, sparse_embedder
 
 