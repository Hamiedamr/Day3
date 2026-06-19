from langgraph.prebuilt import create_react_agent
from langchain_ollama import ChatOllama
from langchain.tools import tool
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from pdf_processor import hybrid_search_rrf


def create_rag_agent(qdrant_client, collection_name, dense_embedder, sparse_embedder):
    @tool(response_format="content_and_artifact")
    def retrieve_context(query: str):
        """Retrieve relevant documents using hybrid search with RRF fusion."""
        docs = hybrid_search_rrf(
            client=qdrant_client,
            collection_name=collection_name,
            query_text=query,
            dense_embedder=dense_embedder,
            sparse_embedder=sparse_embedder,
            limit=5
        )
        print(f"DEBUG: Retrieved {len(docs)} documents for query: {query}")
        serialized = "\n\n".join(
            f"Source: {doc['metadata']}\nContent: {doc['content']}\nScore: {doc['score']}"
            for doc in docs
        )
        return serialized, docs
    system_prompt = """You are a document assistant. You have ONE tool: retrieve_context.

RULE: For EVERY user question, you MUST call retrieve_context FIRST before answering anything.
Do not answer from memory. Do not skip the tool call.

After receiving the tool results, answer the question using only that information.
If the tool returns no relevant information, say "I don't have enough information to answer that."
Cite the source when possible."""

    llm = ChatOllama(model="qwen3:4b-instruct", think=False)

    agent = create_react_agent(
        model=llm,
        tools=[retrieve_context],
        prompt=system_prompt,
        checkpointer=InMemorySaver(),
    )
    return agent


def stream_agent_response(agent, user_query, thread_id="default"):
    inputs = {"messages": [{"role": "user", "content": user_query}]}
    config = {"configurable": {"thread_id": thread_id}}

    for chunk in agent.stream(inputs, stream_mode="values", config=config):
        latest_message = chunk["messages"][-1]

        if isinstance(latest_message, HumanMessage):
            continue
        if isinstance(latest_message, AIMessage):
            if isinstance(latest_message.content, str) and latest_message.content.strip():
                yield latest_message.content
            elif isinstance(latest_message.content, list):
                for block in latest_message.content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        yield block["text"]
        if hasattr(latest_message, 'tool_calls') and latest_message.tool_calls:
            yield f"\nSearching: {[tc['name'] for tc in latest_message.tool_calls]}\n"