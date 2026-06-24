from rag import load_and_split_pdf, setup_qdrant_collection

PDF_FILES = [
    "documents/Abdelazim_Rabie_Backend_NodeJS.pdf",
    "documents/Manar Rabie (UX_UI Designer).pdf",
]


def main():
    all_chunks = []
    for path in PDF_FILES:
        print(f"\n{'=' * 60}")
        print(f"Loading: {path}")
        print("=" * 60)

        chunks = load_and_split_pdf(path)
        print(f"Total chunks: {len(chunks)}")
        all_chunks.extend(chunks)

    print(f"\n{'=' * 60}")
    print(f"Setting up Qdrant collection with {len(all_chunks)} chunks")
    print("=" * 60)

    client, collection_name, dense_embedder, sparse_embedder = setup_qdrant_collection(all_chunks)
    print(f"Collection: {collection_name}")
    print(f"Dense embedder: {dense_embedder.model_name}")
    print(f"Sparse embedder: {sparse_embedder.model_name}")
    print(f"Client: {client}")
    print("\nDone. Check http://localhost:6333/dashboard")


if __name__ == "__main__":
    main()
