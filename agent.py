"""MCP client agent that connects to the RAG MCP server over HTTP.

The agent discovers tools dynamically from the MCP server rather than
importing retrieval functions directly. This enforces the MCP boundary.

- Uses langchain-mcp-adapters + create_agent (2026 syntax)
- Streams responses with stream_mode='values'
"""

from langchain.agents import create_agent          # 2026 syntax
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.checkpoint.memory import InMemorySaver
from langchain.messages import AIMessage

MCP_URL = "http://127.0.0.1:8000/mcp"


# Configure the MCP client to reach your server over HTTP.
# In langchain-mcp-adapters the streamable-HTTP transport value is "http"
# ("streamable_http" is also accepted as an alias). The server side runs
# mcp.run(transport="streamable-http") — same transport, different spelling.
def build_mcp_client():
    """Create a MultiServerMCPClient configured to reach the RAG server over HTTP."""
    return MultiServerMCPClient(
        {
            "rag": {
                "url": MCP_URL,
                "transport": "http",  # streamable-http alias
            }
        }
    )


# Load tools from the MCP server (async) and build the agent.
async def create_rag_agent():
    """Load tools from the MCP server over HTTP and build the agent."""
    client = build_mcp_client()

    # Fetch the MCP tools over HTTP — this is a network call!
    tools = await client.get_tools()

    system_prompt = """
    You are a helpful AI assistant with access to a document knowledge base
    through MCP tools served over HTTP.
    You have NO prior knowledge of the uploaded document — you must use the retrieve_context tool to answer any question about it.

    ## TOOL USAGE RULES (Critical)

    - When the user says retrieve, search, find, look up, list, show, or get information from the document → ALWAYS call retrieve_context. Do NOT try to answer from memory.
    - If the user asks "what are the tasks", "list the steps", "what does the lab contain", or any question about the document's structure or content → ALWAYS call retrieve_context to find the relevant sections.
    - The retrieval uses hybrid search (semantic + keyword) with RRF fusion for best results.

    ## WHEN SEARCH RETURNS NO RESULTS

    If retrieve_context returns "No relevant documents found." or an empty result:
    1. Try 2-3 different search queries with different phrasing. For example, if "lab tasks" returns nothing, try "Task 1", "## Task", or "step by step".
    2. Use the retrieve_context tool with queries that match the document's likely structure (headings, numbered items, keywords).
    3. Only after 3 failed attempts, say "I don't have enough information to answer that question".

    ## OUTPUT RULES

    - Always cite your sources when using retrieved information.
    - When listing structured content (tasks, steps, items), present them in a clear numbered or bulleted format.
    - You can ask follow-up questions if the query is unclear.
    """

    # Create the agent with the MCP-loaded tools (2026 syntax)
    agent = create_agent(
        model="ollama:qwen3:4b-instruct",  # provider:model string
        tools=tools,                         # the tools loaded from the MCP server
        system_prompt=system_prompt,
        checkpointer=InMemorySaver(),
    )

    return agent


async def stream_agent_response(agent, user_query, thread_id="session_001"):
    """Stream the agent response with stream_mode='values' (2026 approach).

    Because MCP tool calls are async, the streamer is async too.
    """
    inputs = {"messages": [{"role": "user", "content": user_query}]}
    config = {"configurable": {"thread_id": thread_id}}

    # Stream with stream_mode="values"
    async for chunk in agent.astream(
        inputs,
        stream_mode="values",     # "values"
        config=config
    ):
        latest_message = chunk["messages"][-1]

        if isinstance(latest_message, AIMessage) and latest_message.content:
            yield latest_message.content
        elif hasattr(latest_message, "tool_calls") and latest_message.tool_calls:
            yield f"\n🔍 Calling MCP tool: {[tc['name'] for tc in latest_message.tool_calls]}\n"
