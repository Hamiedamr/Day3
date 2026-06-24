import os
from rag_core import load_and_split_pdf, setup_qdrant_collection

if __name__ == "__main__":
    file_path = os.path.join("documents", "React Lecture 3.pdf")
    try:
        chunks = load_and_split_pdf(file_path)
        client, collection_name, dense_embedder, sparse_embedder = setup_qdrant_collection(
            chunks, collection_name="rag_demo", dense_model="jinaai/jina-embeddings-v2-base-en"
        )
        collection_info = client.get_collection(collection_name)
        print(f"\nVerification status:")
        print(f"Collection Name: {collection_name}")
        print(f"Points Count: {collection_info.points_count}")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
