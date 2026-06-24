from langchain.agents import create_agent
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.messages import AIMessage


MCP_URL = "http://127.0.0.1:8000/mcp"


def build_mcp_client():
    return MultiServerMCPClient(
        {
            "rag": {
                "url": MCP_URL,
                "transport": "http",
            }
        }
    )


async def create_rag_agent():
    client = build_mcp_client()

    tools = await client.get_tools()

    system_prompt = """
You are a strict RAG assistant connected to a document knowledge base through MCP tools.

CRITICAL RULES:
- For every user question, you MUST call the retrieve_context tool first.
- Never answer from your own knowledge before calling retrieve_context.
- Use only the retrieved context to answer.
- If retrieve_context returns no useful context, say:
  "I don't have enough information to answer that question".
- Always mention the source page/file when available.
"""

    agent = create_agent(
        model="ollama:qwen3:4b-instruct",
        tools=tools,
        system_prompt=system_prompt,
        checkpointer=InMemorySaver(),
    )

    return agent


async def stream_agent_response(agent, user_query, thread_id="session_001"):
    """
    Stream the agent response with stream_mode='values'.
    """
    inputs = {
        "messages": [
            {
                "role": "user",
                "content": user_query
            }
        ]
    }

    config = {
        "configurable": {
            "thread_id": thread_id
        }
    }

    async for chunk in agent.astream(
        inputs,
        stream_mode="values",
        config=config
    ):
        latest_message = chunk["messages"][-1]

        if hasattr(latest_message, "tool_calls") and latest_message.tool_calls:
            yield f"\n🔍 Calling MCP tool: {[tc['name'] for tc in latest_message.tool_calls]}\n"

        elif isinstance(latest_message, AIMessage) and latest_message.content:
            yield latest_message.content