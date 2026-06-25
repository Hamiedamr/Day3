from mcp.server.fastmcp import FastMCP
from rag_core import hybrid_search_rrf

mcp = FastMCP("rag-tools")

@mcp.tool()
def retrieve_context(query: str, limit: int = 5) -> str:
    """Retrieve documents using hybrid search."""
    docs = hybrid_search_rrf(query_text=query, limit=limit)
    if not docs:
        return "No documents found."
    return "\n\n".join(
        f"Source: {doc['metadata']}\nScore: {doc['score']:.4f}\nContent: {doc['content']}"
        for doc in docs
    )

@mcp.tool()
def server_info() -> str:
    """Get server info."""
    return "RAG MCP server is running."

if __name__ == "__main__":
    mcp.run(transport="streamable-http")
