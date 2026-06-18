from langchain_pymupdf4llm import PyMuPDF4LLMLoader
from langchain_text_splitters import MarkdownTextSplitter


def load_and_split_pdf(file_path):
    # TODO: Initialize PyMuPDF4LLMLoader with the given file path
    loader = PyMuPDF4LLMLoader(file_path=file_path)

    docs = loader.load()

    # TODO: Initialize MarkdownTextSplitter with appropriate chunk settings
    splitter = MarkdownTextSplitter(chunk_size=1000, chunk_overlap=200)

    # TODO: Split the documents into chunks
    chunks = splitter.split_documents(docs)

    return chunks
