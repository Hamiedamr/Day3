from langchain_core.messages import AIMessage

def stream_agent_response(agent, user_query, thread_id="default"):
    """Stream the agent response with stream_mode='values' (2026 recommended approach).
    Args:
        agent: The created agent
        user_query: The user's question
        thread_id: Conversation thread ID for persistence
    """
    inputs = {
        "messages": [{"role": "user", "content": user_query}]
    }
    
    config = {
        "configurable": {
            "thread_id": thread_id
        }
    }
    
    for chunk in agent.stream(
        inputs,
        stream_mode="values",
        config=config
    ):
        latest_message = chunk["messages"][-1]
        
        if isinstance(latest_message, AIMessage) and latest_message.content:
            yield latest_message.content
        elif hasattr(latest_message, 'tool_calls') and latest_message.tool_calls:
            yield f"\n🔍 Searching: {[tc['name'] for tc in latest_message.tool_calls]}\n"

if __name__ == "__main__":
    print("--- Running Task 11: Real-time Streaming Workflow ---")
    try:
        from task10 import create_rag_agent
        from qdrant_client import QdrantClient
        from fastembed import TextEmbedding, SparseTextEmbedding
        
        client = QdrantClient(url="http://localhost:6333")
        dense  = TextEmbedding(model_name="jinaai/jina-embeddings-v2-base-en")
        sparse = SparseTextEmbedding(model_name="Qdrant/bm25")
        
        agent_instance = create_rag_agent(client, "rag_hybrid_collection", dense, sparse)
        
        query = "What is FOSS and what licenses are associated with it?"
        print(f"User: {query}\n")
        
        for update in stream_agent_response(agent_instance, query, thread_id="session_streaming_1"):
            print(update, end="", flush=True)
            
    except Exception as e:
        print(f"\nError executing Task 11 streaming: {e}")