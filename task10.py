import sys
import io
from rag_core import create_rag_agent

if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    print("Creating Agentic RAG Agent...")
    try:
        agent = create_rag_agent()
        print("Agent created successfully.")
        query = "What is React?"
        print(f"\nTesting agent with query: '{query}'...")
        inputs = {"messages": [{"role": "user", "content": query}]}
        config = {"configurable": {"thread_id": "test_session_001"}}
        response = agent.invoke(inputs, config=config)
        last_message = response["messages"][-1]
        print(f"\nAgent Response:\n{last_message.content}")
    except Exception as e:
        print(f"\nError: {e}")
        print("Note: Ensure Ollama is running and the model is pulled.")
