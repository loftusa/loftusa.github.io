"""Search endpoint tests — deliberately NOT importing main.py.

chromadb is not installed in the test venv, so rag.retrieve will raise inside
the router's try/except.  We verify graceful degradation (200 + empty results)
and the other contract points (422 on bad input, k clamping).
"""
from unittest.mock import patch

from backend.app.routers.search import router
from fastapi import FastAPI
from fastapi.testclient import TestClient

app = FastAPI()
app.include_router(router)
client = TestClient(app)


def test_graceful_degradation_when_chromadb_missing():
    r = client.get("/search?q=hello")
    assert r.status_code == 200
    body = r.json()
    assert body["query"] == "hello"
    assert body["results"] == []
    assert "error" in body


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
