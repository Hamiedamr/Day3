from qdrant_client import QdrantClient, models
from fastembed import TextEmbedding, SparseTextEmbedding
import uuid

def setup_qdrant_collection(chunks):
    """
    Initializes local Qdrant client, configures a hybrid collection 
    (Dense + Sparse vectors), and uploads document chunks as points.
    """
    client = QdrantClient(url="http://localhost:6333")
    
    dense_embedding_model = "jinaai/jina-embeddings-v2-base-en"
    sparse_embedding_model = "Qdrant/bm25"
    
    dense_embedder = TextEmbedding(model_name=dense_embedding_model)
    sparse_embedder = SparseTextEmbedding(model_name=sparse_embedding_model)
    
    collection_name = "local_rag_collection"
    
    sample_embedding = list(dense_embedder.embed(["sample text"]))[0]
    vector_size = len(sample_embedding)
    
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
    
    points = []
    texts = [chunk.page_content for chunk in chunks]
    
    dense_vectors = list(dense_embedder.embed(texts))
    sparse_vectors = list(sparse_embedder.embed(texts))
    
    for idx, (chunk, dense_vec, sparse_vec) in enumerate(zip(chunks, dense_vectors, sparse_vectors)):
        sparse_indices = sparse_vec.indices.tolist()
        sparse_values = sparse_vec.values.tolist()
        
        point = models.PointStruct(
            id=str(uuid.uuid4()),
            vector={
                "dense": dense_vec.tolist() if hasattr(dense_vec, 'tolist') else list(dense_vec),
                "sparse": models.SparseVector(
                    indices=sparse_indices,
                    values=sparse_values
                )
            },
            payload={
                "page_content": chunk.page_content,
                "metadata": chunk.metadata,
                "chunk_id": idx
            }
        )
        points.append(point)
    
    client.upload_points(
        collection_name=collection_name,
        points=points,
        batch_size=64,
        parallel=2,
        max_retries=3,
        wait=False
    )
    
    return client, collection_name, dense_embedder, sparse_embedder