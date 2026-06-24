import os
import uuid
from qdrant_client import QdrantClient, models
from fastembed import TextEmbedding, SparseTextEmbedding
from task7 import load_and_split_pdf

def setup_qdrant_collection(chunks):
    """Setup Qdrant collection using native client with dense + sparse vectors.
    
    Uses FastEmbed for generating embeddings locally.
    """
    # Initialize Native Qdrant Client
    print("Initializing QdrantClient...")
    client = QdrantClient(url="http://localhost:6333")
    
    # Initialize embedding models (FastEmbed)
    print("Initializing FastEmbed dense and sparse embedding models...")
    dense_embedding_model = "jinaai/jina-embeddings-v2-base-en"
    sparse_embedding_model = "Qdrant/bm25"
    
    dense_embedder = TextEmbedding(model_name=dense_embedding_model)
    sparse_embedder = SparseTextEmbedding(model_name=sparse_embedding_model)
    
    # Create collection with explicit vector configurations
    collection_name = "javascript_guide"
    
    # Get embedding dimensions by encoding a sample text
    sample_embedding = list(dense_embedder.embed(["sample text"]))[0]
    vector_size = len(sample_embedding)
    
    # Create collection if it doesn't exist
    if not client.collection_exists(collection_name=collection_name):
        print(f"Creating Qdrant collection '{collection_name}'...")
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
        print(f"Collection '{collection_name}' successfully created.")
    else:
        print(f"Collection '{collection_name}' already exists.")
    
    # Only upload if the collection is empty
    collection_info = client.get_collection(collection_name)
    if collection_info.points_count == 0:
        print(f"Collection is empty. Generating embeddings for {len(chunks)} chunks and uploading points...")
        # Prepare points for upload with dense + sparse vectors
        points = []
        texts = [chunk.page_content for chunk in chunks]
        
        # Generate embeddings in batches
        dense_vectors = list(dense_embedder.embed(texts))
        sparse_vectors = list(sparse_embedder.embed(texts))
        
        print("Uploading points to Qdrant...")
        for idx, (chunk, dense_vec, sparse_vec) in enumerate(zip(chunks, dense_vectors, sparse_vectors)):
            # Convert sparse vector to Qdrant format
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
        
        # Upload points using native upload_points (with parallelization)
        client.upload_points(
            collection_name=collection_name,
            points=points,
            batch_size=64,
            parallel=2,
            max_retries=3,
            wait=True
        )
        print("Points successfully uploaded.")
    else:
        print(f"Collection already has {collection_info.points_count} points. Skipping upload.")
    
    return client, collection_name, dense_embedder, sparse_embedder

if __name__ == "__main__":
    # Load and split default document first
    file_path = os.path.join("documents", "Introdcution to Javascript.pdf")
    try:
        chunks = load_and_split_pdf(file_path)
        client, collection_name, dense_embedder, sparse_embedder = setup_qdrant_collection(chunks)
        
        # Check final state
        collection_info = client.get_collection(collection_name)
        print(f"\nVerification status:")
        print(f"Collection Name: {collection_name}")
        print(f"Points Count: {collection_info.points_count}")
        print("Task 8 execution completed successfully.")
        
    except Exception as e:
        print(f"\nError running Task 8: {e}")
        import traceback
        traceback.print_exc()
