from langchain.messages import AIMessage, HumanMessage, ToolMessage


def stream_agent_response(agent, user_query, thread_id="default"):
    inputs = {"messages": [{"role": "user", "content": user_query}]}
    config = {"configurable": {"thread_id": thread_id}}

    for chunk in agent.stream(inputs, stream_mode="values", config=config):
        latest_message = chunk["messages"][-1]

        if isinstance(latest_message, AIMessage) and latest_message.content:
            yield latest_message.content
        elif hasattr(latest_message, "tool_calls") and latest_message.tool_calls:
            yield f"\nSearching...: {[tc['name'] for tc in latest_message.tool_calls]}\n"


def stream_with_updates(agent, user_query, thread_id="session_001"):
    config = {"configurable": {"thread_id": thread_id}}

    for chunk in agent.stream(
        {"messages": [{"role": "user", "content": user_query}]},
        stream_mode="values",
        config=config,
    ):
        latest_message = chunk["messages"][-1]

        if isinstance(latest_message, HumanMessage):
            print(f"User: {latest_message.content}")
        elif isinstance(latest_message, AIMessage):
            if latest_message.content:
                print(f"Agent: {latest_message.content}")
            if latest_message.tool_calls:
                print(f"\n[Tool Call] {[tc['name'] for tc in latest_message.tool_calls]}")
        elif isinstance(latest_message, ToolMessage):
            print(f"[Tool Result] {latest_message.content[:200]}...")
