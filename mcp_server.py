from mcp.server.fastmcp import FastMCP
from rag_core import hybrid_search_rrf


mcp = FastMCP("rag-tools")


@mcp.tool()
def retrieve_context(query: str, limit: int = 5) -> str:
    """Retrieve relevant documents from the knowledge base using hybrid search."""
    print(f"\n[retrieve_context] query={query}, limit={limit}")

    docs = hybrid_search_rrf(
        query_text=query,
        limit=limit
    )

    print(f"[retrieve_context] docs_count={len(docs)}")

    if not docs:
        return "No relevant documents found."

    serialized = "\n\n".join(
        f"Source file: {doc['metadata'].get('source')}\n"
        f"Page: {doc['metadata'].get('page')}\n"
        f"Score: {doc['score']:.4f}\n"
        f"Content: {doc['content']}"
        for doc in docs
    )

    print("[retrieve_context] returning context")
    return serialized


@mcp.tool()
def server_info() -> str:
    """
    Return basic info about the RAG MCP server.
    """
    return "RAG MCP server is running. Tool: retrieve_context hybrid RRF search."


if __name__ == "__main__":
    mcp.run(transport="streamable-http")