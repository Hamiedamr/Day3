from langchain.messages import AIMessage


def stream_agent_response(agent, user_query, thread_id="default"):
    """Stream the agent response with stream_mode='values' (2026 recommended approach).

    Args:
        agent: The created agent
        user_query: The user's question
        thread_id: Conversation thread ID for persistence
    """
    # TODO: Prepare the input messages (2026 format)
    inputs = {
        "messages": [{"role": "user", "content": user_query}]
    }

    # TODO: Prepare config with thread_id for conversation memory
    config = {
        "configurable": {
            "thread_id": thread_id  # Required for checkpointer to work
        }
    }

    # TODO: Stream with stream_mode="values" (2026 recommended approach)
    # Each chunk contains the full state at that point
    for chunk in agent.stream(
        inputs,
        stream_mode="values",  # "values" for full state at each step
        config=config,
    ):
        # Access the latest message from the state
        latest_message = chunk["messages"][-1]

        if isinstance(latest_message, AIMessage) and latest_message.content:
            yield latest_message.content
        elif hasattr(latest_message, "tool_calls") and latest_message.tool_calls:
            # Agent is calling a tool
            yield f"\n🔍 Searching: {[tc['name'] for tc in latest_message.tool_calls]}\n"