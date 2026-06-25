"""MCP client + ReAct agent for the Agentic RAG system.

The agent discovers retrieval tools over HTTP from the MCP server.
It must NOT import hybrid_search_rrf directly.
"""
from langchain.agents import create_agent
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.checkpoint.memory import InMemorySaver
from langchain.messages import AIMessage

MCP_URL = "http://127.0.0.1:8000/mcp"
LLM_MODEL = "ollama:qwen3:4b-instruct"


def build_mcp_client():
    """Return an MCP client configured for the HTTP tool server."""
    return MultiServerMCPClient(
        {
            "rag": {
                "url": MCP_URL,
                "transport": "http",
            }
        }
    )


async def create_rag_agent():
    """Connect to the MCP server over HTTP, load tools, and build the agent."""
    client = build_mcp_client()

    # Fetch the MCP tools over HTTP — this is a network call!
    tools = await client.get_tools()

    system_prompt = """You are a helpful AI assistant with access to a document knowledge base
through MCP tools served over HTTP.

Instructions:
- Use the retrieve_context tool when you need information from the documents.
- The retrieval uses hybrid search (semantic + keyword) with RRF fusion.
- Always cite your sources when using retrieved information.
- If the retrieved context doesn't contain relevant information, say
  "I don't have enough information to answer that question".
"""

    agent = create_agent(
        model=LLM_MODEL,
        tools=tools,
        system_prompt=system_prompt,
        checkpointer=InMemorySaver(),
    )

    # Return the client too so the caller can keep the HTTP connection alive.
    return agent, client


async def stream_agent_response(agent, user_query, thread_id="session_001"):
    """Stream the agent response with stream_mode='values'."""
    inputs = {"messages": [{"role": "user", "content": user_query}]}
    config = {"configurable": {"thread_id": thread_id}}

    async for chunk in agent.astream(
        inputs,
        stream_mode="values",
        config=config
    ):
        latest_message = chunk["messages"][-1]

        if isinstance(latest_message, AIMessage) and latest_message.content:
            yield latest_message.content
        elif hasattr(latest_message, "tool_calls") and latest_message.tool_calls:
            yield f"\n🔍 Calling MCP tool: {[tc['name'] for tc in latest_message.tool_calls]}\n"
