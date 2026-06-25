"""Full end-to-end test of the RAG MCP agent."""
import asyncio
import sys
sys.stdout.reconfigure(encoding='utf-8')

from agent import create_rag_agent, stream_agent_response

async def test():
    print("=== FULL E2E TEST: RAG MCP Agent ===\n")
    
    print("1. Creating agent (connecting to MCP server)...")
    agent = await create_rag_agent()
    print("   [OK] Agent created\n")
    
    print("2. Query: 'What tools do you have?'")
    print("   Response: ", end="", flush=True)
    async for token in stream_agent_response(agent, 
        "What tools do you have available?",
        thread_id="e2e_001"):
        print(token, end="", flush=True)
    print("\n")
    
    print("3. Query: 'Search the document for relevant content.'")
    print("   Response: ", end="", flush=True)
    async for token in stream_agent_response(agent, 
        "Search the document for relevant content and summarize what it's about.",
        thread_id="e2e_002"):
        print(token, end="", flush=True)
    print("\n")
    
    print("=== TEST COMPLETE ===")

if __name__ == "__main__":
    asyncio.run(test())
