import sys
from task7 import load_and_split_pdf
from task8 import setup_qdrant_collection


def main():
    if len(sys.argv) < 2:
        print("Usage: python app.py <pdf_file_path>")
        sys.exit(1)

    file_path = sys.argv[1]

    # Step 1: Load and chunk the PDF
    print(f"[1/2] Loading and chunking PDF: {file_path}")
    chunks = load_and_split_pdf(file_path)
    print(f"       -> {len(chunks)} chunks created")

    # Step 2: Index chunks into Qdrant
    print(f"[2/2] Indexing chunks into Qdrant...")
    client, collection_name, dense_embedder, sparse_embedder = setup_qdrant_collection(chunks)
    print(f"       -> Done! Collection '{collection_name}' is ready for hybrid search.")


if __name__ == "__main__":
    main()
