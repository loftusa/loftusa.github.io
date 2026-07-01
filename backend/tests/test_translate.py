"""Translate endpoint tests — exercises the router in isolation (no main.py, no real Anthropic)."""
from __future__ import annotations

from backend.app.routers.translate import router
from fastapi import FastAPI
from fastapi.testclient import TestClient

_app = FastAPI()
_app.include_router(router)
client = TestClient(_app)


def test_empty_text_returns_422():
    r = client.post("/translate", json={"text": ""})
    assert r.status_code == 422


def test_whitespace_only_text_returns_422():
    r = client.post("/translate", json={"text": "   "})
    assert r.status_code == 422


def test_valid_text_returns_200_even_when_anthropic_fails():
    # anthropic is not installed in the test env; the router catches the ImportError
    # and streams an error string — status must be 200 (set before streaming begins)
    r = client.post("/translate", json={"text": "Hello, how are you?"})
    assert r.status_code == 200
