from langchain.agents import create_agent
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver

from task9 import hybrid_search_rrf


def create_rag_agent(qdrant_client, collection_name, dense_embedder, sparse_embedder):
    """Create an agentic RAG agent with native Qdrant hybrid search tool."""

    @tool(response_format="content_and_artifact")
    def retrieve_context(query: str):
        """Retrieve relevant documents using hybrid search with RRF fusion.

        Args:
            query: The search query to find relevant documents
        """
        docs = hybrid_search_rrf(
            client=qdrant_client,
            collection_name=collection_name,
            query_text=query,
            dense_embedder=dense_embedder,
            sparse_embedder=sparse_embedder,
            limit=5,
        )

        serialized = "\n\n".join(
            f"Source: {doc['metadata']}\nContent: {doc['content']}\nScore: {doc['score']}"
            for doc in docs
        )
        return serialized, docs

    system_prompt = """
    You are a helpful AI assistant with access to a document knowledge base.

    Instructions:
    - Use the retrieve_context tool when you need information from the documents
    - The retrieval uses hybrid search (semantic + keyword) with RRF fusion for best results
    - Always cite your sources when using retrieved information
    - If the retrieved context doesn't contain relevant information, say "I don't have enough information to answer that question"
    - You can ask follow-up questions if the query is unclear
    """

    agent = create_agent(
        model="ollama:qwen3:4b-instruct",
        tools=[retrieve_context],
        system_prompt=system_prompt,
        checkpointer=InMemorySaver(),
    )

    return agent
