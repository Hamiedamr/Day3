from langchain.agents import create_agent
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from search_engine import hybrid_search_rrf

def create_rag_agent(qdrant_client, collection_name, dense_embedder, sparse_embedder):
    """
    Creates an intelligent Agentic RAG agent equipped with a native Qdrant hybrid search tool.
    Uses the modern 2026 unified LangChain/LangGraph orchestration system.
    """
    
    @tool(response_format="content_and_artifact")  
    def retrieve_context(query: str):
        """
        Retrieve relevant documents using hybrid search with RRF fusion.
        Args:
            query: The search query to find relevant documents from the knowledge base.
        """
        docs = hybrid_search_rrf(
            client=qdrant_client,
            collection_name=collection_name,
            query_text=query,
            dense_embedder=dense_embedder,
            sparse_embedder=sparse_embedder,
            limit=4  
        )
        
        serialized = "\n\n".join(
            f"Source: {doc['metadata'].get('source', 'Unknown')}\nContent: {doc['content']}\nScore: {doc['score']:.4f}"
            for doc in docs
        )
        return serialized, docs
    
    system_prompt = """You are a helpful AI assistant with access to a document knowledge base.

Instructions:
- Use the retrieve_context tool when you need specialized information from the loaded documents to answer the query.
- The retrieval system utilizes high-performance hybrid search (semantic + keyword) with RRF fusion.
- Always cite your sources explicitly (e.g., filename, page numbers) when using retrieved information.
- If the retrieved context doesn't contain enough relevant facts to confidently answer, clearly say: "I don't have enough information to answer that question."
- You are allowed to ask concise follow-up questions if the user's intent is ambiguous.
"""
    
    memory_checkpointer = InMemorySaver()
    

    agent = create_agent(
        model="ollama:qwen3:4b-instruct",  
        tools=[retrieve_context],          
        system_prompt=system_prompt,       
        checkpointer=memory_checkpointer   
    )
    
    return agent