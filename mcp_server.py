from mcp.server.fastmcp import FastMCP
from rag_core import hybrid_search_rrf

mcp = FastMCP("rag-tools")


@mcp.tool()
def retrieve_context(query: str, limit: int = 5) -> str:
    """Retrieve relevant documents from the knowledge base using hybrid
    search (dense + sparse) with RRF fusion.

    Args:
        query: The natural-language search query.
        limit: Maximum number of chunks to return.
    """
    docs = hybrid_search_rrf(query_text=query, limit=limit)

    if not docs:
        return "No relevant documents found."

    serialized = "\n\n".join(
        f"Source: {doc['metadata']}\nScore: {doc['score']:.4f}\nContent: {doc['content']}"
        for doc in docs
    )
    return serialized


@mcp.tool()
def server_info() -> str:
    """Return basic info about the RAG MCP server."""
    return "RAG MCP server is running. Tool: retrieve_context (hybrid RRF search)."


if __name__ == "__main__":
    mcp.run(transport="streamable-http")