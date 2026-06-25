"""MCP client + ReAct agent for the Agentic RAG system.

The agent discovers retrieval tools over HTTP from the MCP server.
It must NOT import hybrid_search_rrf directly.
"""
from langchain.agents import create_agent
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.checkpoint.memory import InMemorySaver
from langchain.messages import AIMessage
from langchain_ollama import ChatOllama

MCP_URL = "http://127.0.0.1:8000/mcp"
LLM_MODEL = "qwen3:4b-instruct"


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

    system_prompt = """You MUST use the retrieve_context tool for every factual question.
Do NOT answer from your own knowledge. Base your answer strictly on the retrieved context
and cite sources when possible."""

    # NOTE: the README suggests the "ollama:..." model string, but the current
    # langchain create_agent does not bind tools correctly to Ollama via that
    # string. Using ChatOllama directly makes tool calling reliable.
    agent = create_agent(
        model=ChatOllama(model=LLM_MODEL),
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
