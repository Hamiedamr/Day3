"""One-time PDF ingestion script.

Loads a PDF, splits it into chunks, and pushes them into Qdrant via
the shared rag_core logic. Run once per document before querying.

Usage:
    uv run python ingest.py documents/your_file.pdf
"""

import sys
from rag_core import load_and_split_pdf, setup_qdrant_collection


def ingest(file_path, collection_name="documents"):
    """Load a PDF, split into chunks, and push into Qdrant."""
    # Load + split the PDF
    chunks = load_and_split_pdf(file_path)

    # Push chunks into Qdrant (native client)
    name = setup_qdrant_collection(chunks, collection_name=collection_name)

    print(f"Ingested {len(chunks)} chunks into collection '{name}'.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: uv run python ingest.py <path_to_pdf>")
        sys.exit(1)
    ingest(sys.argv[1])
