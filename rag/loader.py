from langchain_pymupdf4llm import PyMuPDF4LLMLoader
from langchain_text_splitters import MarkdownTextSplitter

from rag.config import CHUNK_SIZE, CHUNK_OVERLAP


def load_and_split_pdf(file_path):
    loader = PyMuPDF4LLMLoader(file_path=file_path)
    docs = loader.load()
    splitter = MarkdownTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = splitter.split_documents(docs)
    return chunks
