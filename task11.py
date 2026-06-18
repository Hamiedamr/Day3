from langchain.messages import AIMessage


def _message_content_to_text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                parts.append(item.get("text") or item.get("content") or "")
        return "".join(parts)
    return str(content)


def stream_agent_response(agent, user_query: str, thread_id: str = "default"):
    """
    Stream the agent's response using stream_mode='values'.
    Yields strings that can be concatenated to build the full reply.
    """
    # Input message in the 2026 messages format
    inputs = {
        "messages": [{"role": "user", "content": user_query}]
    }
 
    # Config with thread_id so the checkpointer can maintain memory
    config = {
        "configurable": {
            "thread_id": thread_id   # required for InMemorySaver to work
        }
    }
 
    last_ai_message_id = None
    last_ai_content = ""
    announced_tool_calls = set()

    # stream_mode="values" returns the full state at each agent step.
    for chunk in agent.stream(
        inputs,
        stream_mode="values",   # full state at each agent step
        config=config
    ):
        messages = chunk.get("messages", [])
        if not messages:
            continue

        latest_message = messages[-1]
 
        tool_calls = getattr(latest_message, "tool_calls", None)
        if tool_calls:
            tool_names = []
            for tool_call in tool_calls:
                call_id = tool_call.get("id") or repr(tool_call)
                if call_id in announced_tool_calls:
                    continue
                announced_tool_calls.add(call_id)
                tool_names.append(tool_call.get("name", "tool"))

            if tool_names:
                yield f"\nSearching: {', '.join(tool_names)}\n\n"

        if isinstance(latest_message, AIMessage) and latest_message.content:
            message_id = getattr(latest_message, "id", None)
            content = _message_content_to_text(latest_message.content)

            if message_id != last_ai_message_id:
                last_ai_message_id = message_id
                last_ai_content = ""

            if content.startswith(last_ai_content):
                delta = content[len(last_ai_content):]
            else:
                delta = content

            if delta:
                yield delta
                last_ai_content = content
 
