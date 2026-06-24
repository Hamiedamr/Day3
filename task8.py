import uuid
from qdrant_client import QdrantClient, models
from fastembed import TextEmbedding, SparseTextEmbedding

def setup_qdrant_collection(chunks):
    """Setup Qdrant collection using native client with dense + sparse vectors.
    Uses FastEmbed for generating embeddings locally.
    """
    client = QdrantClient(url="http://localhost:6333", timeout=60)
    
    dense_embedding_model  = "jinaai/jina-embeddings-v2-base-en"
    sparse_embedding_model = "Qdrant/bm25"
    
    dense_embedder  = TextEmbedding(model_name=dense_embedding_model)
    sparse_embedder = SparseTextEmbedding(model_name=sparse_embedding_model)
    
    collection_name = "rag_hybrid_collection"
    
    sample_embedding = list(dense_embedder.embed(["sample text"]))[0]
    vector_size      = len(sample_embedding)

    if not client.collection_exists(collection_name=collection_name):
        print(f"Creating a new hybrid collection: {collection_name}...")
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
    else:
        print(f"Collection '{collection_name}' already exists.")
    
    points = []
    texts  = [chunk.page_content for chunk in chunks]
    
    print("Generating dense and sparse embeddings locally via FastEmbed...")
    dense_vectors  = list(dense_embedder.embed(texts))
    sparse_vectors = list(sparse_embedder.embed(texts))
    
    print("Formatting points into structure...")
    for idx, (chunk, dense_vec, sparse_vec) in enumerate(zip(chunks, dense_vectors, sparse_vectors)):
        sparse_indices = sparse_vec.indices.tolist()
        sparse_values  = sparse_vec.values.tolist()
        
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
                "metadata":     chunk.metadata,
                "chunk_id":     idx
            }
        )
        points.append(point)
    
    print(f"Uploading {len(points)} points to Qdrant database...")
    client.upload_points(
        collection_name=collection_name,
        points=points,
        batch_size=64,
        parallel=2,
        max_retries=3,
        wait=True
    )
    print("Successfully synchronized points!")
    
    return client, collection_name, dense_embedder, sparse_embedder

if __name__ == "__main__":
    try:
        from task7 import load_and_split_pdf
        chunks = load_and_split_pdf("data/SElinux.pdf")
        print(f"🔄 Debug: Type of chunks is {type(chunks)}")
        print(f"🔄 Debug: Number of chunks extracted: {len(chunks)}")
        db_client, col_name, d_emb, s_emb = setup_qdrant_collection(chunks)
        print("\nVerification: Ingestion complete!")
        print(f"Active collection status check count: {db_client.get_collection(col_name).points_count}")
    except Exception as e:
        print(f"Error executing Task 8: {e}")