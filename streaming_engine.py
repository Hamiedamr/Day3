from langchain_core.messages import AIMessage

def stream_agent_response(agent, user_query, thread_id="default"):
    """
    Streams the agentic RAG response token-by-token or step-by-step using 
    the modern 2026 stream_mode='values' state management protocol.
    """
    inputs = {
        "messages": [{"role": "user", "content": user_query}]
    }
    
    config = {
        "configurable": {
            "thread_id": thread_id  
        }
    }
    
    last_seen_length = 0
    
    for chunk in agent.stream(
        inputs,
        stream_mode="values",     
        config=config
    ):
        latest_message = chunk["messages"][-1]
        
        if hasattr(latest_message, 'tool_calls') and latest_message.tool_calls:
            tool_names = [tc['name'] for tc in latest_message.tool_calls]
            yield f"\n [Agent Action] Invoking Knowledge Base Search: {tool_names}\n"
            
        elif isinstance(latest_message, AIMessage) and latest_message.content:
            full_content = latest_message.content
            new_content = full_content[last_seen_length:]
            if new_content:
                yield new_content
                last_seen_length = len(full_content)