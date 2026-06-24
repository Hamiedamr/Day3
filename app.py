from rag import load_and_split_pdf, setup_qdrant_collection, create_rag_agent

PDF_FILES = [
    "documents/Abdelazim_Rabie_Backend_NodeJS.pdf",
    "documents/Manar Rabie (UX_UI Designer).pdf",
]

TEST_QUERY = "What backend technologies does Abdelazim know?"


def main():
    all_chunks = []
    for path in PDF_FILES:
        chunks = load_and_split_pdf(path)
        print(f"{path}: {len(chunks)} chunks")
        all_chunks.extend(chunks)

    print(f"\nSetting up Qdrant collection with {len(all_chunks)} chunks...")
    client, collection_name, dense_embedder, sparse_embedder = setup_qdrant_collection(all_chunks)
    print(f"Collection ready: {collection_name}")

    print("\nCreating agentic RAG agent...")
    agent = create_rag_agent(client, collection_name, dense_embedder, sparse_embedder)
    print(f"Agent ready: {agent}")

    print(f"\n{'=' * 60}")
    print(f"Query: {TEST_QUERY}")
    print("=" * 60)

    result = agent.invoke(
        {"messages": [{"role": "user", "content": TEST_QUERY}]},
        config={"configurable": {"thread_id": "test_001"}},
    )

    print(f"\n--- Messages trace ({len(result['messages'])} messages) ---")
    for i, msg in enumerate(result["messages"]):
        kind = type(msg).__name__
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            names = [tc["name"] for tc in msg.tool_calls]
            print(f"[{i}] {kind}: tool_calls={names}")
        elif hasattr(msg, "content") and msg.content:
            preview = msg.content[:150].replace("\n", " ")
            print(f"[{i}] {kind}: {preview}{'...' if len(msg.content) > 150 else ''}")
        else:
            print(f"[{i}] {kind}: (empty)")

    final = result["messages"][-1]
    print(f"\n--- Final answer ---\n{final.content}")


if __name__ == "__main__":
    main()
