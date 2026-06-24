import os
from document_processor import load_and_split_pdf
from vdb_setup import setup_qdrant_collection
from rag_agent import create_rag_agent
from streaming_engine import stream_agent_response

def main():
    pdf_path = os.path.join("documents", "sample.pdf")
    
    if not os.path.exists(pdf_path):
        print(f" Error: Place a PDF at '{pdf_path}' first.")
        return

    print("Step 1: Loading and splitting PDF...")
    chunks = load_and_split_pdf(pdf_path)
    print(f"Success: Split into {len(chunks)} chunks.")

    print("\nStep 2: Initializing Qdrant and uploading hybrid vectors...")
    client, collection_name, dense_embedder, sparse_embedder = setup_qdrant_collection(chunks)
    print("Success: Chunks indexed successfully.")

    print("\nStep 3: Creating Agentic RAG Agent...")
    agent_executor = create_rag_agent(client, collection_name, dense_embedder, sparse_embedder)
    print("Success: Agent configured.")

    print("\n Agent is ready with live Streaming capabilities!")
    user_query = "Summarize the key takeaways of this document and cite the source."
    print(f"\nUser: {user_query}")
    print("AI Response: ", end="", flush=True)
    
    for token in stream_agent_response(agent_executor, user_query, thread_id="session_1"):
        print(token, end="", flush=True)
    print("\n\n Stream Finished.")

if __name__ == "__main__":
    main()