from langchain_core.messages import AIMessage

def stream_agent_response(agent, user_query, thread_id="default"):
    """Streams token events and internal tracking updates from the model state."""
    inputs = {"messages": [{"role": "user", "content": user_query}]}
    config = {"configurable": {"thread_id": thread_id}}
    
    for chunk in agent.stream(inputs, stream_mode="values", config=config):
        latest_message = chunk["messages"][-1]
        
        if isinstance(latest_message, AIMessage) and latest_message.content:
            yield latest_message.content
        elif hasattr(latest_message, 'tool_calls') and latest_message.tool_calls:
            tool_names = [tc['name'] for tc in latest_message.tool_calls]
            yield f"\nSearching knowledge base using tools: {tool_names}\n"