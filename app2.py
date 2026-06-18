import argparse

from task7 import load_and_split_pdf
from task8 import setup_qdrant_collection
from task9 import hybrid_search_rrf


def main():
    parser = argparse.ArgumentParser(
        description="Index a PDF into Qdrant and run hybrid RRF search."
    )
    parser.add_argument("pdf_file_path", help="Path to the PDF file to index")
    parser.add_argument("query", help="Search query to run against the indexed chunks")
    parser.add_argument(
        "--limit", type=int, default=5, help="Number of search results to return"
    )
    args = parser.parse_args()

    print(f"[1/3] Loading and chunking PDF: {args.pdf_file_path}")
    chunks = load_and_split_pdf(args.pdf_file_path)
    print(f"       -> {len(chunks)} chunks created")

    print("[2/3] Indexing chunks into Qdrant")
    client, collection_name, dense_embedder, sparse_embedder = setup_qdrant_collection(
        chunks
    )
    print(f"       -> Collection '{collection_name}' is ready")

    print(f"[3/3] Running hybrid search: {args.query}")
    documents = hybrid_search_rrf(
        client,
        collection_name,
        args.query,
        dense_embedder,
        sparse_embedder,
        limit=args.limit,
    )

    for index, document in enumerate(documents, start=1):
        content = document["content"].replace("\n", " ").strip()
        preview = content[:300] + ("..." if len(content) > 300 else "")
        print(f"\n[{index}] score={document['score']:.4f} id={document['id']}")
        print(preview)
        if document["metadata"]:
            print(f"metadata={document['metadata']}")


if __name__ == "__main__":
    main()
