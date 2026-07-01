"""Search endpoint tests — deliberately NOT importing main.py.

When chromadb is absent from the venv, rag.retrieve raises inside the router's
try/except and we can verify graceful degradation (200 + empty results + error
key). When chromadb IS installed (it arrived in the dev deps), that failure
path isn't reachable this way, so the degradation test skips itself rather
than asserting an environment-dependent outcome. Other contract points (422 on
bad input, k clamping) hold either way.
"""
import importlib.util
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.app.routers.search import router

app = FastAPI()
app.include_router(router)
client = TestClient(app)

_CHROMADB_INSTALLED = importlib.util.find_spec("chromadb") is not None


@pytest.mark.skipif(
    _CHROMADB_INSTALLED,
    reason="chromadb installed — the import-failure degradation path isn't reachable",
)
def test_graceful_degradation_when_chromadb_missing():
    r = client.get("/search?q=hello")
    assert r.status_code == 200
    body = r.json()
    assert body["query"] == "hello"
    assert body["results"] == []
    assert "error" in body


def test_graceful_degradation_when_retrieve_raises():
    """Same contract, environment-independent: any rag.retrieve failure degrades
    to 200 + empty results + error key (never a 500 to the site)."""
    with patch(
        "backend.app.routers.search.rag.retrieve", side_effect=RuntimeError("boom")
    ):
        r = client.get("/search?q=hello")
    assert r.status_code == 200
    body = r.json()
    assert body["results"] == [] and "error" in body


def test_missing_q_returns_422():
    r = client.get("/search")
    assert r.status_code == 422


def test_empty_q_returns_422():
    r = client.get("/search?q=")
    assert r.status_code == 422


def test_whitespace_q_returns_422():
    # %20%20 = two spaces; strip() makes it empty
    r = client.get("/search?q=%20%20")
    assert r.status_code == 422


def test_k_clamped_to_20():
    fake = [
        {"source": f"s{i}", "text": f"t{i}", "distance": float(i)} for i in range(30)
    ]
    with patch("backend.app.services.rag.retrieve", return_value=fake):
        r = client.get("/search?q=hello&k=100")
    assert r.status_code == 200
    assert len(r.json()["results"]) == 20


def test_k_clamped_to_1():
    fake = [{"source": "s0", "text": "t0", "distance": 0.1}]
    with patch("backend.app.services.rag.retrieve", return_value=fake):
        r = client.get("/search?q=hello&k=0")
    assert r.status_code == 200
    assert len(r.json()["results"]) == 1
