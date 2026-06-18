from qdrant_client import models


def _point_to_document(point):
    payload = point.payload or {}
    return {
        "content": payload.get("page_content", ""),
        "metadata": payload.get("metadata", {}),
        "score": getattr(point, "score", None),
        "id": point.id,
        "chunk_id": payload.get("chunk_id"),
    }


def _dedupe_documents(documents):
    seen = set()
    unique_documents = []

    for document in documents:
        metadata = document.get("metadata") or {}
        key = (
            metadata.get("source") or metadata.get("file_path"),
            metadata.get("page"),
            document.get("chunk_id"),
            document.get("content"),
        )
        if key in seen:
            continue

        seen.add(key)
        unique_documents.append(document)

    return unique_documents


def _fetch_neighbor_chunks(client, collection_name, documents, window=2):
    neighbor_docs = []
    chunk_ids = {
        doc.get("chunk_id")
        for doc in documents
        if isinstance(doc.get("chunk_id"), int)
    }

    for chunk_id in sorted(chunk_ids):
        start = max(0, chunk_id - window)
        stop = chunk_id + window + 1

        for neighbor_id in range(start, stop):
            points, _ = client.scroll(
                collection_name=collection_name,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="chunk_id",
                            match=models.MatchValue(value=neighbor_id),
                        )
                    ]
                ),
                limit=5,
                with_payload=True,
                with_vectors=False,
            )
            neighbor_docs.extend(_point_to_document(point) for point in points)

    return neighbor_docs


def hybrid_search_rrf(
    client,
    collection_name,
    query_text,
    dense_embedder,
    sparse_embedder,
    limit=8,
    prefetch_limit=60,
    neighbor_window=2,
):
    """
    Retrieve documents using dense + sparse prefetch with Reciprocal Rank
    Fusion (RRF), then include nearby chunks for more complete context.
    """
    dense_query = list(dense_embedder.embed([query_text]))[0]
    sparse_query = list(sparse_embedder.embed([query_text]))[0]

    sparse_indices = sparse_query.indices.tolist()
    sparse_values = sparse_query.values.tolist()

    results = client.query_points(
        collection_name=collection_name,
        prefetch=[
            models.Prefetch(
                query=models.SparseVector(
                    indices=sparse_indices,
                    values=sparse_values,
                ),
                using="sparse",
                limit=prefetch_limit,
            ),
            models.Prefetch(
                query=(
                    dense_query.tolist()
                    if hasattr(dense_query, "tolist")
                    else list(dense_query)
                ),
                using="dense",
                limit=prefetch_limit,
            ),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        with_payload=True,
        limit=limit,
    )

    documents = [_point_to_document(point) for point in results.points]

    if neighbor_window:
        documents.extend(
            _fetch_neighbor_chunks(
                client=client,
                collection_name=collection_name,
                documents=documents,
                window=neighbor_window,
            )
        )

    max_documents = limit + (limit * max(neighbor_window, 0) * 2)
    return _dedupe_documents(documents)[:max_documents]
