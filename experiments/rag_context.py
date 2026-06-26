"""
Shared RAG retrieval module.

Loads a ChromaDB collection and retrieves relevant chunks for a query.
Used by both chat_api.py (production) and evals/rag_eval.py (evaluation).

No CLI/rich/click dependencies — importable from anywhere.
"""

from pathlib import Path

import chromadb

DEFAULT_CHROMA_DIR = Path(__file__).parent / "rag" / "data" / "chroma_rag"


def load_collection(
    chroma_dir: Path = DEFAULT_CHROMA_DIR,
) -> chromadb.Collection:
    """Load the persistent Chroma collection. Raises if empty."""
    assert chroma_dir.exists(), f"Chroma directory not found: {chroma_dir}"
    client = chromadb.PersistentClient(path=str(chroma_dir))
    collection = client.get_or_create_collection(name="resume_rag")
    assert collection.count() > 0, (
        f"Chroma collection is empty at {chroma_dir}. "
        "Run experiments/rag/ex3_retrieval.py to build the index."
    )
    return collection


def retrieve_context(
    collection: chromadb.Collection,
    query: str,
    n_results: int = 5,
    max_distance: float = 1.3,
) -> list[dict]:
    """Retrieve relevant chunks for a query.

    Returns list of {source, text, distance} dicts, filtered by max_distance.
    """
    results = collection.query(query_texts=[query], n_results=n_results)

    chunks = []
    for doc, dist, meta in zip(
        results["documents"][0],
        results["distances"][0],
        results["metadatas"][0],
    ):
        if dist > max_distance:
            continue
        source = meta.get("title", meta.get("source_url", "?"))
        chunks.append({"source": source, "text": doc, "distance": dist})
    return chunks


def format_rag_context(chunks: list[dict]) -> str | None:
    """Format retrieved chunks into a system message string.

    Returns None if no chunks are provided.
    """
    if not chunks:
        return None

    context_block = "\n\n---\n\n".join(
        f"[Source: {c['source']}]\n{c['text']}" for c in chunks
    )
    return (
        "The following are additional context chunks retrieved from "
        "Alex's papers, talks, and projects. Use them to give more "
        "specific answers when relevant.\n\n" + context_block
    )
