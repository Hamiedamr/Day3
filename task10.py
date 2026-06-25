from langchain.agents import create_agent
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver

@tool(response_format="content_and_artifact")
def retrieve_context(query: str, client, collection_name, dense_embedder, sparse_embedder):
    """Retrieve relevant documents using hybrid search with RRF fusion."""
    from task9 import hybrid_search_rrf
    docs = hybrid_search_rrf(client, collection_name, query, dense_embedder, sparse_embedder, limit=5)
    serialized = "\n\n".join(
        f"Source: {doc['metadata']}\nContent: {doc['content']}\nScore: {doc['score']}"
        for doc in docs
    )
    return serialized, docs

def create_rag_agent(client, collection_name, dense_embedder, sparse_embedder):
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
        tools=[],
        system_prompt=system_prompt,
        checkpointer=InMemorySaver(),
    )

    return agent

if __name__ == "__main__":
    from qdrant_client import QdrantClient
    from fastembed import TextEmbedding, SparseTextEmbedding

    client = QdrantClient(url="http://localhost:6333", check_compatibility=False)
    collection_name = "agentic_rag_collection"
    dense_embedder = TextEmbedding(model_name="jinaai/jina-embeddings-v2-base-en")
    sparse_embedder = SparseTextEmbedding(model_name="Qdrant/bm25")

    agent = create_rag_agent(client, collection_name, dense_embedder, sparse_embedder)
    query = "What is asthma and how is it predicted?"
    config = {"configurable": {"thread_id": "session_001"}}
    for chunk in agent.stream(
        {"messages": [{"role": "user", "content": query}]},
        stream_mode="values",
        config=config
    ):
        latest_message = chunk["messages"][-1]
        if hasattr(latest_message, "content") and latest_message.content:
            print(latest_message.content, end="", flush=True)
