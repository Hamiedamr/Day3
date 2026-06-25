from langchain_core.messages import AIMessage


async def stream_agent_response(agent, user_query, thread_id="session_001"):
    """Stream the agent response asynchronously (2026 approach, MCP version)."""
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


if __name__ == "__main__":
    import asyncio
    from agent import create_rag_agent

    async def main():
        agent = await create_rag_agent()
        async for token in stream_agent_response(agent, "What university did this person graduate from?", thread_id="test_thread"):
            print(token, end="", flush=True)
        print()

    asyncio.run(main())