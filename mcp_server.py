from mcp.server.fastmcp import FastMCP
from qdrant_client import QdrantClient
from fastembed import TextEmbedding, SparseTextEmbedding
from rag_core import hybrid_search_rrf

mcp = FastMCP("rag-tools")

_client = QdrantClient(url="http://localhost:6333")
_collection_name = "rag_demo"
_dense_embedder = TextEmbedding(model_name="jinaai/jina-embeddings-v2-base-en")
_sparse_embedder = SparseTextEmbedding(model_name="Qdrant/bm25")


@mcp.tool()
def retrieve_context(query: str, limit: int = 5) -> str:
    docs = hybrid_search_rrf(_client, _collection_name, query, _dense_embedder, _sparse_embedder, limit)
    if not docs:
        return "No relevant documents found."
    return "\n\n".join(
        f"Source: {doc['metadata']}\nScore: {doc['score']:.4f}\nContent: {doc['content']}"
        for doc in docs
    )


@mcp.tool()
def server_info() -> str:
    return "RAG MCP server is running. Tool: retrieve_context (hybrid RRF search)."


if __name__ == "__main__":
    mcp.run(transport="streamable-http")
