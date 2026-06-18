from langchain_pymupdf4llm import PyMuPDF4LLMLoader
from langchain_text_splitters import MarkdownTextSplitter


def load_and_split_pdf(file_path: str):
    # Initialize the loader with the given file path
    loader = PyMuPDF4LLMLoader(file_path=file_path)
 
    # Load the document pages
    docs = loader.load()
 
    # Split into chunks — 1 000 chars with 200-char overlap keeps context intact
    splitter = MarkdownTextSplitter(chunk_size=1000, chunk_overlap=200)
 
    # Split the documents into chunks
    chunks = splitter.split_documents(docs)
    


    return chunks


