from langchain.agents import create_agent
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langchain.messages import AIMessage
from search import hybrid_search_rrf

def create_rag_agent(qdrant_client, collection_name, dense_embedder, sparse_embedder):
    # define retrieval tool
    @tool(response_format="content_and_artifact")
    def retrieve_context(query: str):
        """Retrieve relevant documents using hybrid search."""
        docs = hybrid_search_rrf(
            client=qdrant_client,
            collection_name=collection_name,
            query_text=query,
            dense_embedder=dense_embedder,
            sparse_embedder=sparse_embedder,
            limit=5
        )
        
        serialized = "\n\n".join(
            f"Source: {doc['metadata']}\nContent: {doc['content']}\nScore: {doc['score']}"
            for doc in docs
        )
        return serialized, docs
    
    # set system instructions
    system_prompt = """
    You are a helpful AI assistant with document access.
    - use retrieve_context when needed
    - always cite your sources
    - say if info is missing
    """
    
    agent = create_agent(
        model="ollama:qwen3:4b-instruct",
        tools=[retrieve_context],
        system_prompt=system_prompt,
        checkpointer=InMemorySaver(),
    )
    
    return agent

def stream_agent_response(agent, user_query, thread_id="default"):
    # prepare messages and config
    inputs = {"messages": [{"role": "user", "content": user_query}]}
    config = {"configurable": {"thread_id": thread_id}}
    
    # stream with values mode
    for chunk in agent.stream(inputs, stream_mode="values", config=config):
        msg = chunk["messages"][-1]       
        if isinstance(msg, AIMessage) and msg.content:
            yield msg.content
        elif hasattr(msg, 'tool_calls') and msg.tool_calls:
            yield f"\n searching: {[tc['name'] for tc in msg.tool_calls]}\n"
