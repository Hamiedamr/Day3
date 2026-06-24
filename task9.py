import sys
import io
from qdrant_client import QdrantClient
from fastembed import TextEmbedding, SparseTextEmbedding
from rag_core import hybrid_search_rrf


def hybrid_search_demo(query_text, limit=5):
    client = QdrantClient(url="http://localhost:6333")
    collection_name = "rag_demo"
    dense_embedder = TextEmbedding(model_name="jinaai/jina-embeddings-v2-base-en")
    sparse_embedder = SparseTextEmbedding(model_name="Qdrant/bm25")
    return hybrid_search_rrf(client, collection_name, query_text, dense_embedder, sparse_embedder, limit)


if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    query = sys.argv[1] if len(sys.argv) > 1 else "What is React?"
    print(f"Executing hybrid search for query: '{query}'...")

    try:
        results = hybrid_search_demo(query, limit=3)
        print(f"\nSuccessfully retrieved {len(results)} results:")
        for idx, doc in enumerate(results):
            print(f"\n--- Result {idx + 1} (Score: {doc['score']:.4f}) ---")
            print(f"Source: {doc['metadata'].get('source', 'N/A')}, Page: {doc['metadata'].get('page', 'N/A')}")
            print("-" * 50)
            print(doc['content'][:400] + ("\n..." if len(doc['content']) > 400 else ""))
            print("-" * 50)
    except Exception as e:
        print(f"Error: {e}")
