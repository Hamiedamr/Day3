from qdrant_client import QdrantClient
from fastembed import TextEmbedding, SparseTextEmbedding
from agent import create_rag_agent

def run_agentic_pipeline():
    print("Connecting to Qdrant and loading models...")
    client = QdrantClient(url="http://localhost:6333")
    collection_name = "nesma_cv_collection"
    
    dense_embedder = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
    sparse_embedder = SparseTextEmbedding(model_name="Qdrant/bm25")
    
    print("Initializing the RAG Agent...")
    agent = create_rag_agent(
        qdrant_client=client,
        collection_name=collection_name,
        dense_embedder=dense_embedder,
        sparse_embedder=sparse_embedder
    )
    
    user_question = "What frameworks is Nesma experienced with, and when did she graduate?"
    print(f"\nQuestion: '{user_question}'\n")
    
    config = {"configurable": {"thread_id": "session_1"}}
    
    events = list(agent.stream({"messages": [{"role": "user", "content": user_question}]}, config))
    
    print("\nFinal Agent Response:")
    if events:
        last_message = events[-1].get("messages", []) or events[-1].get("agent", {}).get("messages", [])
        if last_message:
            print(last_message[-1].content)
        else:
            for event in events:
                if "agent" in event and "messages" in event["agent"]:
                    print(event["agent"]["messages"][-1].content)

if __name__ == "__main__":
    run_agentic_pipeline()