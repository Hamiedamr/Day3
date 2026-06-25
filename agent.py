from langchain.agents import create_agent
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.checkpoint.memory import InMemorySaver
from langchain.messages import AIMessage

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

async def create_rag_agent(checkpointer=None):
    client = build_mcp_client()
    tools = await client.get_tools()
    system_prompt = "You must always call the retrieve_context tool to search for information before answering any user question. Never answer without calling retrieve_context first."
    return create_agent(
        model="ollama:qwen3:4b-instruct",
        tools=tools,
        system_prompt=system_prompt,
        checkpointer=checkpointer or InMemorySaver(),
    )

async def stream_agent_response(agent, user_query, thread_id="session_001"):
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
            yield f"\nCalling MCP tool: {[tc['name'] for tc in latest_message.tool_calls]}\n"
