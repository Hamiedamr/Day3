# agent.py
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_mcp_adapters.client import MultiServerMCPClient

MCP_URL = "http://127.0.0.1:8000/mcp"


def build_mcp_client():
    return MultiServerMCPClient(
        {
            "rag": {
                "url": MCP_URL,
                "transport": "http",
            }
        }
    )


async def create_rag_agent():
    """Create a RAG pipeline that always calls the MCP retrieve_context tool
    before answering.

    qwen3:4b-instruct does not reliably decide to call tools via create_agent,
    whether the tool is local or MCP-based. This pipeline removes that
    uncertainty: it always calls the MCP tool first, then asks the LLM to
    answer using the retrieved context. The agent still goes through MCP
    over HTTP — it just doesn't rely on the model's judgment to trigger it.
    """
    client = build_mcp_client()
    tools = await client.get_tools()

    # Find the retrieve_context tool from the MCP-loaded tools
    retrieve_tool = next(t for t in tools if t.name == "retrieve_context")

    llm = ChatOllama(model="qwen3:4b-instruct", temperature=0)

    system_prompt = """You are a helpful AI assistant answering questions about an uploaded document.

Use ONLY the provided context below to answer the question.
If the context doesn't contain the answer, say "I don't have enough information to answer that question."
Be concise and cite the relevant part of the context when possible."""

    class MCPRagAgent:
        def __init__(self, llm, system_prompt, retrieve_tool):
            self.llm = llm
            self.system_prompt = system_prompt
            self.retrieve_tool = retrieve_tool

        async def _build_messages(self, user_message):
            # Always call the MCP tool over HTTP — this is the actual MCP boundary
            context = await self.retrieve_tool.ainvoke({"query": user_message, "limit": 3})

            full_prompt = f"""Context from the document:
{context}

Question: {user_message}

Answer using only the context above."""
            return [
                SystemMessage(content=self.system_prompt),
                HumanMessage(content=full_prompt)
            ], context

        async def ainvoke(self, inputs, config=None):
            user_message = inputs["messages"][-1]["content"]
            messages, context = await self._build_messages(user_message)
            response = await self.llm.ainvoke(messages)
            return {
                "messages": inputs["messages"] + [{"role": "assistant", "content": response.content}],
                "retrieved_context": context
            }

        async def astream(self, inputs, stream_mode="values", config=None):
            user_message = inputs["messages"][-1]["content"]

            # Signal that retrieval is happening, mimicking a tool-call indicator
            yield {"messages": inputs["messages"] + [
                AIMessage(content="", tool_calls=[{"name": "retrieve_context", "args": {"query": user_message}, "id": "auto", "type": "tool_call"}])
            ]}

            messages, context = await self._build_messages(user_message)
            response = await self.llm.ainvoke(messages)

            yield {"messages": inputs["messages"] + [AIMessage(content=response.content)]}

    return MCPRagAgent(llm, system_prompt, retrieve_tool)


if __name__ == "__main__":
    import asyncio
    import time

    async def main():
        agent = await create_rag_agent()
        config = {"configurable": {"thread_id": "test_thread"}}

        print("Invoking agent...")
        start = time.time()

        response = await agent.ainvoke(
            {"messages": [{"role": "user", "content": "What university did this person graduate from?"}]},
            config=config
        )

        elapsed = time.time() - start
        print(f"\nDone in {elapsed:.1f} seconds")
        print("\n=== FINAL ANSWER ===")
        print(response["messages"][-1]["content"])
        print("\n=== RETRIEVED CONTEXT (via MCP) ===")
        print(response["retrieved_context"][:300])

    asyncio.run(main())