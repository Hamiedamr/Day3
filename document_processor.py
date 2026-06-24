from langchain_pymupdf4llm import PyMuPDF4LLMLoader
from langchain_text_splitters import MarkdownTextSplitter

def load_and_split_pdf(file_path):
    """
    Loads a PDF document, converts it to Markdown format, 
    and splits it into chunks for vector search.
    """
    loader = PyMuPDF4LLMLoader(file_path=file_path)
    docs = loader.load()
    
    splitter = MarkdownTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    
    return chunks