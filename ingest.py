import sys
from rag_core import load_and_split_pdf, setup_qdrant_collection

def ingest(file_path, collection_name="agentic_rag_collection"):
    chunks = load_and_split_pdf(file_path)
    name = setup_qdrant_collection(chunks, collection_name=collection_name)
    print(f"Ingested {len(chunks)} chunks into collection '{name}'.")

if __name__ == "__main__":
    ingest(sys.argv[1])