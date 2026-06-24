from langchain.agents import create_agent
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from task9_search import search_vault  
def create_rag_agent(dense_embedder, sparse_embedder):
    """Creates an agent armed with the custom document search tool."""
    
    @tool(response_format="content_and_artifact")
    def retrieve_context(query: str):
        """Retrieve relevant documents using hybrid search with RRF fusion.
        Args:
            query: The search query to find relevant documents
        """
        docs = search_vault(
            query_text=query,
            dense_embedder=dense_embedder,
            sparse_embedder=sparse_embedder,
            limit=3
        )
        serialized = "\n\n".join(
            f"Source: {doc['metadata']}\nContent: {doc['content']}\nScore: {doc['score']}"
            for doc in docs
        )
        return serialized, docs

    system_prompt = """You are a professional research assistant.
    CRITICAL RULE: You MUST ALWAYS call the 'retrieve_context' tool for every single user query to inspect the database before answering. Do not attempt to answer from your own knowledge.
    Instructions:
    - Use the retrieve_context tool to find information.
    - Always cite your sources when using retrieved information.
    - If the retrieved context doesn't contain relevant information, say 'I don't have enough information to answer that question'.
    """
    
    agent = create_agent(
        model="ollama:qwen3:4b-instruct",
        tools=[retrieve_context],
        system_prompt=system_prompt,
        checkpointer=InMemorySaver(),
    )
    return agent