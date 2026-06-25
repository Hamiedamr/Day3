"""Test Qdrant retrieval and MCP tool directly."""
import asyncio
import sys
sys.stdout.reconfigure(encoding='utf-8')

from rag_core import hybrid_search_rrf

# 1. Test Qdrant retrieval directly
print("=== 1. Direct Qdrant retrieval ===")
docs = hybrid_search_rrf(query_text="JavaScript Node.js web development", limit=3)
print(f"Found {len(docs)} documents")
for d in docs:
    content = d["content"][:150].replace("\n", " ")
    print(f"  Score: {d['score']:.4f} | Content: {content}...")
print()

# 2. Test via MCP tool
print("=== 2. MCP tool test ===")
from langchain_mcp_adapters.client import MultiServerMCPClient
from mcp.types import CallToolResult

async def test_mcp_tool():
    client_mcp = MultiServerMCPClient({
        "rag": {
            "url": "http://127.0.0.1:8000/mcp",
            "transport": "http",
        }
    })
    tools = await client_mcp.get_tools()
    print(f"Loaded {len(tools)} MCP tools")
    
    # Try calling retrieve_context via the LangChain tool
    print("\nCalling retrieve_context tool...")
    result = await tools[0].ainvoke({"query": "JavaScript Node.js web development"})
    print(f"Result ({len(str(result))} chars):")
    print(str(result)[:500])

asyncio.run(test_mcp_tool())
