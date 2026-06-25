from langchain_pymupdf4llm import PyMuPDF4LLMLoader
from langchain_text_splitters import MarkdownTextSplitter


def load_and_split_pdf(file_path):
    """Load a PDF and split it into markdown-aware chunks."""

    loader = PyMuPDF4LLMLoader(file_path=file_path)

    docs = loader.load()

    splitter = MarkdownTextSplitter(chunk_size=1000, chunk_overlap=200)

    chunks = splitter.split_documents(docs)

    return chunks


if __name__ == "__main__":
    chunks = load_and_split_pdf("documents/Mahmoud_Ramadan_Abbas_CV.pdf")
    print(f"Total chunks: {len(chunks)}")
    print(chunks[0].page_content[:300])
    print(chunks[0].metadata)