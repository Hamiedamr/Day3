from langchain_pymupdf4llm import PyMuPDF4LLMLoader
from langchain_text_splitters import MarkdownTextSplitter

def load_and_split_pdf(file_path):
    # Initialize PyMuPDF4LLMLoader with the given file path
    loader = PyMuPDF4LLMLoader(file_path=file_path)
    
    # Load the document
    docs = loader.load()

    splitter = MarkdownTextSplitter(chunk_size=1000, chunk_overlap=200)
    
    # Split the documents into chunks
    chunks = splitter.split_documents(docs)
    
    return chunks
if __name__ == "__main__":
    sample_pdf = "data/SElinux.pdf" 
    
    print("--- Running Task 7: Loading and Splitting Document ---")
    try:
        document_chunks = load_and_split_pdf(sample_pdf)
        print(f"Successfully split document into {len(document_chunks)} chunks.")
        
        for i in range(min(3, len(document_chunks))):
            print(f"\n=========================================")
            print(f" CHUNK {i+1} FULL CONTENT")
            print(f"=========================================")
            print(document_chunks[i].page_content)
            print(f"=========================================\n")
            
    except Exception as e:
        print(f"Error executing Task 7: {e}")