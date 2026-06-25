import sys
from task10 import create_rag_agent

def stream_agent_response(agent, user_query, thread_id="default"):
    from langchain.messages import AIMessage, HumanMessage
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
            yield f"\n--- Searching: {latest_message.tool_calls[0]['name']} ---\n"

if __name__ == "__main__":
    agent = create_rag_agent()
    query = sys.argv[1] if len(sys.argv) > 1 else "What is asthma and how is it predicted?"
    for token in stream_agent_response(agent, query, thread_id="session_001"):
        print(token, end="", flush=True)
    print()
