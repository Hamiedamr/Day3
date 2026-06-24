import os
from rag_core import load_and_split_pdf

if __name__ == "__main__":
    file_path = os.path.join("documents", "React Lecture 3.pdf")
    chunks = load_and_split_pdf(file_path)
    print(f"\n--- Chunk Previews (showing first 3 of {len(chunks)} chunks) ---")
    for idx, chunk in enumerate(chunks[:3]):
        print(f"\nChunk {idx + 1} ({len(chunk.page_content)} characters):")
        print("-" * 50)
        print(chunk.page_content)
        print("-" * 50)
