"""MCP server exposing RAG retrieval tools over HTTP.

This server runs as a standalone process and exposes retrieval tools via
FastMCP over the streamable-http transport. The agent connects to this
server as an MCP client and calls these tools remotely.

Run with:
    uv run python mcp_server.py
"""

from mcp.server.fastmcp import FastMCP
from rag_core import hybrid_search_rrf

# Create the MCP server instance and give it a name
mcp = FastMCP("rag-tools")


# Register the retrieval tool with the @mcp.tool() decorator.
# The docstring + type hints become the tool schema the agent sees.
@mcp.tool()
def retrieve_context(query: str, limit: int = 5) -> str:
    """Retrieve relevant documents from the knowledge base using hybrid
    search (dense + sparse) with RRF fusion.

    Args:
        query: The natural-language search query.
        limit: Maximum number of chunks to return.
    """
    # Call your hybrid search function from rag_core
    docs = hybrid_search_rrf(query_text=query, limit=limit)

    # Format the retrieved documents into a single string for the LLM
    if not docs:
        return "No relevant documents found."

    serialized = "\n\n".join(
        f"Source: {doc['metadata']}\nScore: {doc['score']:.4f}\nContent: {doc['content']}"
        for doc in docs
    )
    return serialized


# Add a second tool, e.g. a health/info tool (optional)
@mcp.tool()
def server_info() -> str:
    """Return basic info about the RAG MCP server."""
    return (
        "RAG MCP server is running. "
        "Tool: retrieve_context (hybrid RRF search with dense + sparse vectors)."
    )


if __name__ == "__main__":
    # Run the server over HTTP using the streamable-http transport.
    # FastMCP serves on http://127.0.0.1:8000/mcp by default.
    mcp.run(transport="streamable-http")
