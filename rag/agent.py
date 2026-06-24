from langchain.agents import create_agent
from langchain.tools import tool
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import InMemorySaver

from rag.config import OLLAMA_MODEL, SEARCH_LIMIT
from rag.search import hybrid_search_rrf


def create_rag_agent(qdrant_client, collection_name, dense_embedder, sparse_embedder):
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
            limit=SEARCH_LIMIT,
        )

        serialized = "\n\n".join(
            f"Source: {doc['metadata']}\nContent: {doc['content']}\nScore: {doc['score']}"
            for doc in docs
        )
        return serialized, docs

    system_prompt = """
    You are a helpful AI assistant with access to a document knowledge base.

    Instructions:
    - ALWAYS call the retrieve_context tool first for any question about documents,
      people, technologies, experience, or any factual content that could be in the knowledge base.
      Do not answer from your own knowledge — retrieve first, then answer.
    - The retrieval uses hybrid search (semantic + keyword) with RRF fusion for best results.
    - Always cite your sources when using retrieved information.
    - Only say "I don't have enough information to answer that question" AFTER you have called
      retrieve_context and the returned context is genuinely irrelevant.
    - You can ask follow-up questions if the query is unclear.
    - For general knowledge questions (math, definitions, small talk) that are NOT about the
      documents, you may answer directly without retrieving.
    """

    agent = create_agent(
        model=ChatOllama(model=OLLAMA_MODEL, temperature=0),
        tools=[retrieve_context],
        system_prompt=system_prompt,
        checkpointer=InMemorySaver(),
    )

    return agent
