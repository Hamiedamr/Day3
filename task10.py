from langchain_ollama import ChatOllama
from langchain.tools import tool
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import InMemorySaver
from task9 import hybrid_search_rrf

def create_rag_agent(qdrant_client, collection_name, dense_embedder, sparse_embedder):
    """Create an agentic RAG agent with native Qdrant hybrid search tool."""
    
    @tool(response_format="content_and_artifact")
    def retrieve_context(query: str):
        """Retrieve relevant documents using hybrid search with RRF fusion."""
        docs = hybrid_search_rrf(query_text=query, limit=5)
            
        serialized = "\n\n".join(
            f"Source: {doc['metadata']}\nContent: {doc['content']}\nScore: {doc['score']}"
            for doc in docs
        )
        return serialized, docs
    
    # /no_think disables qwen3 internal monologue that causes it to skip tool calls
    system_prompt = """/no_think
You are an advanced, context-driven AI Assistant. You have access to a specialized knowledge base through your tools.

Instructions:
1. For ANY question asked by the user, ALWAYS invoke the `retrieve_context` tool immediately to extract factual context from the uploaded document.
2. Analyze the retrieved information carefully to find definitions, explanations, or acronyms relevant to the user's query.
3. Base your final response entirely on the extracted tool results. Always clearly cite or reference the relevant headers or source details present in the context.
4. Do not assume or guess information using external generic knowledge outside the scope of the document's domain.
5. If the tool returns empty results or the context does not contain any relevant information to answer the question accurately, reply exactly with: "I don't have enough information in the provided documentation to answer that question."
"""
    
    # create_agent from langchain.agents uses "ollama:model" string
    # but it maps internally to ChatOllama — we use ChatOllama directly for reliability
    llm = ChatOllama(
        model="qwen3:4b-instruct",
        temperature=0,
        num_ctx=4096,
    )
    
    agent = create_react_agent(
        model=llm,
        tools=[retrieve_context],
        prompt=system_prompt,
        checkpointer=InMemorySaver(),
    )
    
    return agent

if __name__ == "__main__":
    print("--- Running Task 10: Building Agentic RAG Harness ---")
    try:
        from qdrant_client import QdrantClient
        from fastembed import TextEmbedding, SparseTextEmbedding
        
        client = QdrantClient(url="http://localhost:6333")
        dense  = TextEmbedding(model_name="jinaai/jina-embeddings-v2-base-en")
        sparse = SparseTextEmbedding(model_name="Qdrant/bm25")
        
        rag_agent = create_rag_agent(client, "rag_hybrid_collection", dense, sparse)
        print("Agent successfully built!")
        
        config   = {"configurable": {"thread_id": "session_1"}}
        response = rag_agent.invoke(
            {"messages": [{"role": "user", "content": "What is FOSS?"}]},
            config=config
        )
        
        print("\n--- Agent Response ---")
        print(response["messages"][-1].content)
        
    except Exception as e:
        print(f"Error executing Task 10: {e}")