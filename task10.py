import inspect

from langchain.agents import create_agent
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver

from task9 import hybrid_search_rrf


def create_rag_agent(
    qdrant_client,
    collection_name,
    dense_embedder,
    sparse_embedder,
    model="ollama:qwen3:4b-instruct",
):
    """
    Build a ReAct-style agent that decides when to call the hybrid-search
    retrieval tool and streams its reasoning back to the UI.
    """
 
    # ── Retrieval tool ─────────────────────────────────────────────────────────
    @tool(response_format="content_and_artifact")
    def retrieve_context(query: str):
        """Retrieve relevant documents using hybrid search with RRF fusion.
 
        Args:
            query: The search query to find relevant documents.
        """
        docs = hybrid_search_rrf(
            client=qdrant_client,
            collection_name=collection_name,
            query_text=query,
            dense_embedder=dense_embedder,
            sparse_embedder=sparse_embedder,
            limit=5
        )
 
        if not docs:
            return "No relevant documents were found.", []

        # Format retrieved docs into a compact, citation-friendly string.
        serialized_docs = []
        for index, doc in enumerate(docs, start=1):
            metadata = doc.get("metadata", {})
            source = metadata.get("source") or metadata.get("file_path") or "unknown"
            title = metadata.get("title") or "Untitled document"
            page = metadata.get("page", "unknown")
            serialized_docs.append(
                "\n".join(
                    [
                        f"[{index}] Source: {source}",
                        f"Title: {title}",
                        f"Page: {page}",
                        f"Score: {doc.get('score')}",
                        f"Content: {doc.get('content', '')}",
                    ]
                )
            )
        serialized = "\n\n".join(serialized_docs)
        return serialized, docs   # (content, artifact) tuple
 
    # ── System prompt ──────────────────────────────────────────────────────────
    system_prompt = """
You are a helpful AI assistant with access to a document knowledge base.

Instructions:
- Use the retrieve_context tool when you need information from the documents.
- The retrieval uses hybrid search (semantic + keyword) with RRF fusion for best results.
- When retrieved content answers the question, answer from that content and cite Source/Page.
- If the retrieved context does not contain enough relevant information, say "I don't have enough information to answer that question".
- You can ask follow-up questions if the query is unclear.
""".strip()
 
    # ── Agent creation (2026 syntax) ───────────────────────────────────────────
    # "ollama:<model>" — no separate ChatOllama import needed
    create_agent_params = inspect.signature(create_agent).parameters
    prompt_key = (
        "system_prompt"
        if "system_prompt" in create_agent_params
        else "prompt"
        if "prompt" in create_agent_params
        else None
    )
    if prompt_key is None:
        raise TypeError("Installed LangChain create_agent has no prompt parameter.")

    agent = create_agent(
        model=model,                         # provider:model string
        tools=[retrieve_context],            # our hybrid-search tool
        checkpointer=InMemorySaver(),        # persist conversation across turns
        **{prompt_key: system_prompt},
    )
 
    return agent
 
