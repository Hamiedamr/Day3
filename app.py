from rag import load_and_split_pdf, setup_qdrant_collection, hybrid_search_rrf

PDF_FILES = [
    "documents/Abdelazim_Rabie_Backend_NodeJS.pdf",
    "documents/Manar Rabie (UX_UI Designer).pdf",
]

TEST_QUERIES = [
    "Node.js backend developer experience",
    "UX/UI designer dashboards",
    "experience in Egypt",
]


def main():
    all_chunks = []
    for path in PDF_FILES:
        chunks = load_and_split_pdf(path)
        print(f"{path}: {len(chunks)} chunks")
        all_chunks.extend(chunks)

    print(f"\nSetting up Qdrant collection with {len(all_chunks)} chunks...")
    client, collection_name, dense_embedder, sparse_embedder = setup_qdrant_collection(all_chunks)
    print(f"Collection ready: {collection_name}\n")

    for query in TEST_QUERIES:
        print(f"\n{'=' * 60}")
        print(f"Query: {query}")
        print("=" * 60)

        docs = hybrid_search_rrf(
            client=client,
            collection_name=collection_name,
            query_text=query,
            dense_embedder=dense_embedder,
            sparse_embedder=sparse_embedder,
            limit=5,
        )

        print(f"Results: {len(docs)}")
        for i, doc in enumerate(docs):
            source = doc["metadata"].get("file_path", "?")
            preview = doc["content"][:120].replace("\n", " ")
            print(f"  [{i}] score={doc['score']:.4f} | source={source}")
            print(f"      {preview}...")


if __name__ == "__main__":
    main()
