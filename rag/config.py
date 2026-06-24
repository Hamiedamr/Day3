# Chunking (Task 7)
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# Qdrant (Task 8)
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "rag_collection"
DENSE_EMBED_MODEL = "jinaai/jina-embeddings-v2-base-en"
SPARSE_EMBED_MODEL = "Qdrant/bm25"

# Ollama / Agent (Task 10)
OLLAMA_MODEL = "ollama:qwen3:4b-instruct"
SEARCH_LIMIT = 5
