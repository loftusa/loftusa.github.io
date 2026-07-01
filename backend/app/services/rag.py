"""RAG retrieval over the prebaked Chroma collection. Lazy like llm.py: chromadb is only imported
the first time a chat actually needs context, so importing the app stays cheap and test-friendly.
"""
from __future__ import annotations

from functools import lru_cache


@lru_cache(maxsize=1)
def _collection():
    from experiments.rag_context import load_collection

    return load_collection()


def retrieve(query: str) -> list[dict]:
    from experiments.rag_context import retrieve_context

    return retrieve_context(_collection(), query)


def format_context(chunks: list[dict]) -> str:
    from experiments.rag_context import format_rag_context

    return format_rag_context(chunks)


def count() -> int:
    return _collection().count()


def warmup() -> None:
    """Pre-load the embedding model + collection (called at startup to avoid cold-start latency)."""
    retrieve("warmup query")
