from langchain_pymupdf4llm import PyMuPDF4LLMLoader
from langchain_text_splitters import MarkdownTextSplitter
from qdrant_client import QdrantClient, models
from fastembed import TextEmbedding, SparseTextEmbedding
import uuid


def load_and_split_pdf(file_path):
    loader = PyMuPDF4LLMLoader(file_path=file_path)
    docs = loader.load()
    splitter = MarkdownTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    return chunks


def setup_qdrant_collection(chunks):
    client = QdrantClient(url="http://localhost:6333")

    dense_embedder = TextEmbedding(model_name="jinaai/jina-embeddings-v2-base-en")
    sparse_embedder = SparseTextEmbedding(model_name="Qdrant/bm25")

    collection_name = "my_documents"

    sample_embedding = list(dense_embedder.embed(["sample text"]))[0]
    vector_size = len(sample_embedding)

    if not client.collection_exists(collection_name=collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config={
                "dense": models.VectorParams(size=vector_size, distance=models.Distance.COSINE)
            },
            sparse_vectors_config={
                "sparse": models.SparseVectorParams(index=models.SparseIndexParams(on_disk=False))
            }
        )

    texts = [chunk.page_content for chunk in chunks]
    dense_vectors = list(dense_embedder.embed(texts))
    sparse_vectors = list(sparse_embedder.embed(texts))

    points = []
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

    client.upload_points(
        collection_name=collection_name,
        points=points,
        batch_size=64,
        parallel=2,
        max_retries=3,
        wait=True
    )
    print(f"DEBUG: Successfully uploaded {len(points)} chunks to Qdrant.")
    return client, collection_name, dense_embedder, sparse_embedder


def hybrid_search_rrf(client, collection_name, query_text, dense_embedder, sparse_embedder, limit=5):
    dense_query = list(dense_embedder.embed([query_text]))[0]
    sparse_query = list(sparse_embedder.embed([query_text]))[0]

    results = client.query_points(
        collection_name=collection_name,
        prefetch=[
            models.Prefetch(
                query=models.SparseVector(
                    indices=sparse_query.indices.tolist(),
                    values=sparse_query.values.tolist()
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
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        with_payload=True,
        limit=limit
    )

    documents = []
    for point in results.points:
        documents.append({
            "content": point.payload["page_content"],
            "metadata": point.payload["metadata"],
            "score": point.score,
            "id": point.id
        })
    return documents