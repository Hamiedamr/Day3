"""One-time ingestion script: load a PDF and index it into Qdrant."""
import sys

from rag_core import load_and_split_pdf, setup_qdrant_collection


def ingest(file_path, collection_name="documents"):
    """Load + split the PDF and push chunks into Qdrant."""
    chunks = load_and_split_pdf(file_path)
    name = setup_qdrant_collection(chunks, collection_name=collection_name)
    print(f"Ingested {len(chunks)} chunks into collection '{name}'.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: uv run python ingest.py <path_to_pdf>")
        sys.exit(1)
    ingest(sys.argv[1])
