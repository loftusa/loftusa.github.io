"""Semantic search endpoint backed by the Chroma RAG collection.

Gracefully degrades (200, empty results) when chromadb / the collection is
unavailable, so the test suite doesn't need chromadb installed.
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from backend.app.services import rag

router = APIRouter(tags=["search"])


@router.get("/search")
def search(q: str, k: int = 5) -> dict:
    if not q.strip():
        raise HTTPException(status_code=422, detail="q must not be empty")
    k = max(1, min(k, 20))
    try:
        results = rag.retrieve(q)[:k]
    except Exception:
        return {"query": q, "results": [], "error": "search unavailable"}
    return {"query": q, "results": results}
