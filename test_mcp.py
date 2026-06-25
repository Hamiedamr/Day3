"""Test the RAG MCP server directly."""
import asyncio
import httpx

async def test():
    print("=== Testing RAG MCP Server ===\n")
    
    async with httpx.AsyncClient() as client:
        # Test tools/list
        print("1. Testing tools/list...")
        payload = {"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}}
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        }
        resp = await client.post(
            "http://127.0.0.1:8000/mcp",
            json=payload,
            headers=headers,
            timeout=10,
        )
        print(f"   Status: {resp.status_code}")
        print(f"   Response: {resp.text[:500]}")
        print()

        # Test with langchain-mcp-adapters
        print("2. Testing via langchain_mcp_adapters...")
        from langchain_mcp_adapters.client import MultiServerMCPClient
        
        client_mcp = MultiServerMCPClient({
            "rag": {
                "url": "http://127.0.0.1:8000/mcp",
                "transport": "http",
            }
        })
        
        try:
            tools = await client_mcp.get_tools()
            print(f"   Got {len(tools)} tools:")
            for t in tools:
                print(f"     - {t.name}: {t.description[:80]}")
        except Exception as e:
            print(f"   Error: {e}")

if __name__ == "__main__":
    asyncio.run(test())
