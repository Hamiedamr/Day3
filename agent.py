from langchain.agents import create_agent
from langchain.messages import AIMessage
from langchain_ollama import ChatOllama
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.checkpoint.memory import InMemorySaver

from rag.config import OLLAMA_MODEL

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
    You are a helpful AI assistant with access to a document knowledge base
    through MCP tools served over HTTP.

    Instructions:
    - ALWAYS call the retrieve_context tool first for any question about documents,
      people, technologies, experience, or any factual content that could be in the knowledge base.
      Do not answer from your own knowledge — retrieve first, then answer.
    - The retrieval uses hybrid search (semantic + keyword) with RRF fusion.
    - Always cite your sources when using retrieved information.
    - Only say "I don't have enough information to answer that question" AFTER you have called
      retrieve_context and the returned context is genuinely irrelevant.
    - You can ask follow-up questions if the query is unclear.
    - For general knowledge questions (math, definitions, small talk) that are NOT about the
      documents, you may answer directly without retrieving.
    """

    agent = create_agent(
        model=ChatOllama(model=OLLAMA_MODEL, temperature=0),
        tools=tools,
        system_prompt=system_prompt,
        checkpointer=InMemorySaver(),
    )

    return agent


async def stream_agent_response(agent, user_query, thread_id="session_001"):
    inputs = {"messages": [{"role": "user", "content": user_query}]}
    config = {"configurable": {"thread_id": thread_id}}

    async for chunk in agent.astream(inputs, stream_mode="values", config=config):
        latest_message = chunk["messages"][-1]

        if isinstance(latest_message, AIMessage) and latest_message.content:
            yield latest_message.content
        elif hasattr(latest_message, "tool_calls") and latest_message.tool_calls:
            yield f"\n🔍 Calling MCP tool: {[tc['name'] for tc in latest_message.tool_calls]}\n"
