import streamlit as st
import os
from qdrant_client import QdrantClient
from app import load_and_split_pdf
from task9 import search_documents
from task10 import create_rag_agent
from task11 import stream_agent_response

st.set_page_config(page_title="Agentic RAG System", layout="wide")
st.title("🤖 Agentic RAG System with Native Qdrant")

if "agent" not in st.session_state:
    st.session_state.agent = None
if "qdrant_client" not in st.session_state:
    st.session_state.qdrant_client = None
if "collection_name" not in st.session_state:
    st.session_state.collection_name = None
if "dense_embedder" not in st.session_state:
    st.session_state.dense_embedder = None
if "sparse_embedder" not in st.session_state:
    st.session_state.sparse_embedder = None

with st.sidebar:
    st.header("📄 Document Upload")
    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"], key="pdf_uploader")

if uploaded_file is not None:
    temp_path = os.path.join("documents", uploaded_file.name)
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    with st.spinner("Processing document..."):
        chunks = load_and_split_pdf(temp_path)
        from task9 import search_documents

        client = QdrantClient(url="http://localhost:6333", check_compatibility=False)
        collection_name = "agentic_rag_collection"
        dense_embedder = __import__("fastembed").TextEmbedding(model_name="jinaai/jina-embeddings-v2-base-en")

        if not client.collection_exists(collection_name=collection_name):
            from qdrant_client import models
            sample_embedding = list(dense_embedder.embed(["sample text"]))[0]
            vector_size = len(sample_embedding)
            client.create_collection(
                collection_name=collection_name,
                vectors_config={"dense": models.VectorParams(size=vector_size, distance=models.Distance.COSINE)},
                sparse_vectors_config={"sparse": models.SparseVectorParams(index=models.SparseIndexParams(on_disk=False))}
            )

        from fastembed import SparseTextEmbedding
        sparse_embedder = SparseTextEmbedding(model_name="Qdrant/bm25")
        texts = [chunk.page_content for chunk in chunks]
        dense_vectors = list(dense_embedder.embed(texts))
        sparse_vectors = list(sparse_embedder.embed(texts))

        import uuid
        from qdrant_client import models as qdrant_models
        points = []
        for idx, (chunk, dense_vec, sparse_vec) in enumerate(zip(chunks, dense_vectors, sparse_vectors)):
            sparse_indices = sparse_vec.indices.tolist()
            sparse_values = sparse_vec.values.tolist()
            point = qdrant_models.PointStruct(
                id=str(uuid.uuid4()),
                vector={
                    "dense": dense_vec.tolist() if hasattr(dense_vec, "tolist") else list(dense_vec),
                    "sparse": qdrant_models.SparseVector(indices=sparse_indices, values=sparse_values)
                },
                payload={"page_content": chunk.page_content, "metadata": chunk.metadata, "chunk_id": idx}
            )
            points.append(point)

        client.upload_points(collection_name=collection_name, points=points, batch_size=64, parallel=2, max_retries=3, wait=False)

        st.session_state.qdrant_client = client
        st.session_state.collection_name = collection_name
        st.session_state.dense_embedder = dense_embedder
        st.session_state.sparse_embedder = sparse_embedder
        st.session_state.agent = create_rag_agent(client, collection_name, dense_embedder, sparse_embedder)

    st.success("Document processed with hybrid search (RRF fusion)! Agent is ready.")

user_input = st.chat_input("Ask a question about your document...", key="chat_input")

if user_input and st.session_state.agent:
    st.chat_message("user").write(user_input)
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""
        for token in stream_agent_response(
            st.session_state.agent,
            user_input,
            thread_id="session_001"
        ):
            full_response += token
            response_placeholder.markdown(full_response + "▌")
        response_placeholder.markdown(full_response)

elif user_input and not st.session_state.agent:
    st.warning("Please upload a document first!")
