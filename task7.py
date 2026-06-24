import sys
import os
from langchain_pymupdf4llm import PyMuPDF4LLMLoader
from langchain_text_splitters import MarkdownTextSplitter

# Task 7: Document Loading & Text Splitting.
def load_and_split_pdf(file_path):

    # Initialize PyMuPDF4LLMLoader with the given file path
    loader = PyMuPDF4LLMLoader(file_path=file_path)
    print(f"\n successfully intiliazed")
    
    # Load the document
    docs = loader.load()

    
    # Initialize MarkdownTextSplitter with appropriate chunk settings
    splitter = MarkdownTextSplitter(chunk_size=1000, chunk_overlap=200)
    print(f"\n successfully split ")

    # Split the documents into chunks
    chunks = splitter.split_documents(docs)
    print(f"\n successfully chunks")
    
    return chunks

if __name__ == "__main__":
    file_path = os.path.join("documents", "Introdcution to Javascript.pdf")
    chunks = load_and_split_pdf(file_path)
    
    print(f"\n--- Chunk Previews (showing first 3 of {len(chunks)} chunks) ---")
    for idx, chunk in enumerate(chunks[:3]):
        print(f"\nChunk {idx + 1} ({len(chunk.page_content)} characters):")
        print("-" * 50)
        print(chunk.page_content)
        print("-" * 50)
