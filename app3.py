from task7 import load_and_split_pdf
from task8 import setup_qdrant_collection
from task9 import hybrid_search_rrf
from langchain.chat_models import init_chat_model

pdf_file_path = "documents/assets_pdfs_Student Handbook for ITI Programs.pdf"

chunks = load_and_split_pdf(pdf_file_path)
client, collection_name, dense_embedder, sparse_embedder = setup_qdrant_collection(
    chunks
)

# Retrieve context using hybrid search
query = "What is the minimum passing score required for exams, and what happens if a student fails in two courses?"
docs = hybrid_search_rrf(
    client, collection_name, query, dense_embedder, sparse_embedder, limit=5
)

context = "\n\n".join(
    f"Source: {doc['metadata']}\nContent: {doc['content']}"
    for doc in docs
)

model = init_chat_model("ollama:qwen3:4b-instruct")

system_prompt = f"""You are a helpful AI assistant with access to a document knowledge base.

Retrieved context from the Student Handbook:
{context}

Instructions:
- Answer the user's question based ONLY on the retrieved context above.
- If the context doesn't contain relevant information, say "I don't have enough information to answer that question."
- Cite the specific source details from the context.

User question: {query}
"""

# Stream the response token by token
for chunk in model.stream([
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": query},
]):
    if chunk.content:
        print(chunk.content, end="", flush=True)

print()