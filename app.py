from rag import load_and_split_pdf, setup_qdrant_collection, create_rag_agent, stream_with_updates

PDF_FILES = [
    "documents/Abdelazim_Rabie_Backend_NodeJS.pdf",
    "documents/Manar Rabie (UX_UI Designer).pdf",
]

TEST_QUERY = "What backend technologies does Abdelazim know?"


def main():
    all_chunks = []
    for path in PDF_FILES:
        chunks = load_and_split_pdf(path)
        print(f"{path}: {len(chunks)} chunks", flush=True)
        all_chunks.extend(chunks)

    print(f"\nSetting up Qdrant collection with {len(all_chunks)} chunks...", flush=True)
    client, collection_name, dense_embedder, sparse_embedder = setup_qdrant_collection(all_chunks)
    print(f"Collection ready: {collection_name}", flush=True)

    print("\nCreating agentic RAG agent...", flush=True)
    agent = create_rag_agent(client, collection_name, dense_embedder, sparse_embedder)
    print(f"Agent ready\n", flush=True)

    print(f"{'=' * 60}")
    print(f"Streaming query: {TEST_QUERY}")
    print("=" * 60, flush=True)

    stream_with_updates(agent, TEST_QUERY, thread_id="test_001")


if __name__ == "__main__":
    main()
