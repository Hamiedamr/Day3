# agent.py
from langchain.agents import create_agent
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.messages import AIMessage, HumanMessage

MCP_URL = "http://127.0.0.1:8000/mcp"


def build_mcp_client():
    return MultiServerMCPClient(
        {
            "rag": {
                "url": MCP_URL,
                "transport": "streamable_http",
            }
        }
    )


async def create_rag_agent():
    client = build_mcp_client()

    tools = await client.get_tools()
    print(f"DEBUG: tools: {tools}")
    system_prompt = """
    You are an assistant for Sohyla Gomaa. 
    You have access to her CV/Portfolio via the 'retrieve_context' tool.
    
    INSTRUCTIONS:
    - If the user asks about Education, search for terms like 'Bachelor', 'University', 'Faculty', 'Computer Science', 'ITI', or 'Education'.
    - If the user asks about Skills, search for 'technical skills', 'programming', 'frameworks', 'languages', or 'tools'.
    - If the user asks about Experience or Work, search for 'experience', 'work', 'job', 'company', 'position', or 'role'.
    - If the user asks about Projects, search for 'project', 'portfolio', 'built', 'developed', or 'implemented'.
    - If the user asks about Contact info, search for 'email', 'phone', 'linkedin', 'github', or 'contact'.
    - ALWAYS call 'retrieve_context' with a query that matches these professional terms.
    - Base your answers ONLY on the retrieved context. Do not make up information.
    """
   
   

    agent = create_agent(
        model="ollama:qwen3:4b-instruct",
        tools=tools,
        system_prompt=system_prompt,
        checkpointer=InMemorySaver(),
    )

    return agent
#14
async def stream_agent_response(agent, user_query, thread_id="session_001"):
    """Stream the agent response with stream_mode='values' (2026 approach)."""
    inputs = {"messages": [{"role": "user", "content": user_query}]}
    config = {"configurable": {"thread_id": thread_id}}

    last_content = ""
    
    async for chunk in agent.astream(
        inputs,
        stream_mode="values",
        config=config
    ):
        latest_message = chunk["messages"][-1]

        if isinstance(latest_message, HumanMessage):
            continue

        # Show tool call info
        if isinstance(latest_message, AIMessage) and latest_message.tool_calls:
            tool_names = [tc['name'] for tc in latest_message.tool_calls]
            yield f"🔍 Calling MCP tool: {tool_names}\n\n"
            continue

        # Show the final AI response (only the new part)
        if isinstance(latest_message, AIMessage) and latest_message.content:
            new_content = latest_message.content
            if new_content != last_content:
                # Only yield the delta (new portion)
                if new_content.startswith(last_content):
                    yield new_content[len(last_content):]
                else:
                    yield new_content
                last_content = new_content