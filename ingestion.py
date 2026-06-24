import uuid
from qdrant_client import QdrantClient, models
from fastembed import TextEmbedding, SparseTextEmbedding
from langchain_pymupdf4llm import PyMuPDF4LLMLoader
from langchain_text_splitters import MarkdownTextSplitter

def load_and_split_pdf(file_path):
    loader = PyMuPDF4LLMLoader(file_path=file_path)
    docs = loader.load()    
    splitter = MarkdownTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(docs)
    return chunks

def setup_qdrant_collection(chunks):
    # init qdrant
    client = QdrantClient(url="http://localhost:6333")
    
    # init models
    dense_model = "jinaai/jina-embeddings-v2-base-en"
    sparse_model = "Qdrant/bm25"
    
    dense_embedder = TextEmbedding(model_name=dense_model)
    sparse_embedder = SparseTextEmbedding(model_name=sparse_model)
    
    collection_name = "agentic_rag_collection"
    
    # get dimensions
    sample = list(dense_embedder.embed(["sample"]))[0]
    vector_size = len(sample)
    
    # create collection
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
    
    # generate vectors
    dense_vectors = list(dense_embedder.embed(texts))
    sparse_vectors = list(sparse_embedder.embed(texts))
    
    for idx, (chunk, dense_vec, sparse_vec) in enumerate(zip(chunks, dense_vectors, sparse_vectors)):
        point = models.PointStruct(
            id=str(uuid.uuid4()),
            vector={
                "dense": dense_vec.tolist() if hasattr(dense_vec, 'tolist') else list(dense_vec),
                "sparse": models.SparseVector(
                    indices=sparse_vec.indices.tolist(),
                    values=sparse_vec.values.tolist()
                )
            },
            payload={
                "page_content": chunk.page_content,
                "metadata": chunk.metadata,
                "chunk_id": idx
            }
        )
        points.append(point)
    
    # upload points
    client.upload_points(
        collection_name=collection_name,
        points=points,
        batch_size=64,
        parallel=2,
        max_retries=3,
        wait=False
    )
    
    return client, collection_name, dense_embedder, sparse_embedder
