import sys
from langchain.agents import create_agent
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from task9 import hybrid_search_rrf

def create_rag_agent():
    """Create an agentic RAG agent with native Qdrant hybrid search tool."""
    
    # Define the retrieval tool using native Qdrant hybrid search
    @tool(response_format="content_and_artifact")
    def retrieve_context(query: str) -> tuple[str, list]:
        """Retrieve relevant documents using hybrid search with RRF fusion.
        
        Args:
            query: The search query to find relevant documents
        """
        # Call hybrid_search_rrf from task9 (only needs query_text and limit)
        docs = hybrid_search_rrf(query_text=query, limit=5)
        
        # Format the retrieved documents
        serialized = "\n\n".join(
            f"Source: {doc['metadata']}\nContent: {doc['content']}\nScore: {doc['score']}"
            for doc in docs
        )
        return serialized, docs
    
    # Define the system prompt
    system_prompt = """You are a specialized search agent.
CRITICAL: You do not know anything about JavaScript by default. You MUST use the retrieve_context tool to search for information before you can answer ANY user question. Do not answer from your own knowledge. Always query the retrieve_context tool first.

Whenever you answer, you MUST cite the specific page number(s) from the retrieved source (e.g. "according to page 7...") so the user knows where the information came from."""

    # Create the agent using create_agent (2026 syntax)
    agent = create_agent(
        model="ollama:qwen3:4b-instruct",
        tools=[retrieve_context],
        system_prompt=system_prompt,
        checkpointer=InMemorySaver(),
    )
    
    return agent

if __name__ == "__main__":
    # Force UTF-8 output on Windows to support PDF characters
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    
    print("Creating Agentic RAG Agent...")
    try:
        agent = create_rag_agent()
        print("Agent created successfully.")
        
        # Simple test query
        query = "What is JavaScript?"
        print(f"\nTesting agent with query: '{query}'...")
        
        inputs = {
            "messages": [{"role": "user", "content": query}]
        }
        config = {
            "configurable": {
                "thread_id": "test_session_001"
            }
        }
        
        # Invoke the agent
        response = agent.invoke(inputs, config=config)
        
        # Print the last message from the assistant
        last_message = response["messages"][-1]
        print(f"\nAgent Response:\n{last_message.content}")
        
    except Exception as e:
        print(f"\nError creating or running agent: {e}")
        print("Note: If Ollama is not running locally or the model is not pulled, the agent invocation step will fail.")
