"""Translate endpoint tests — exercises the router in isolation (no main.py, no real Anthropic).
Covers the abuse guards too: input-size cap + a per-IP rate limit whose key is scoped so it
can't drain the /corrections budget."""
from __future__ import annotations

from backend.app import config
from backend.app.db import SessionLocal
from backend.app.models import RateLimitHit
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


def test_oversize_text_returns_422():
    r = client.post("/translate", json={"text": "x" * (config.MAX_TRANSLATE_CHARS + 1)})
    assert r.status_code == 422


def test_rate_limited_after_budget_exhausted():
    limit, _ = config.TRANSLATE_RATE
    for _ in range(limit):
        assert client.post("/translate", json={"text": "hola"}).status_code == 200
    assert client.post("/translate", json={"text": "hola"}).status_code == 429


def test_rate_limit_key_is_scoped_away_from_corrections():
    client.post("/translate", json={"text": "hola"})
    s = SessionLocal()
    try:
        keys = {row.client_ip for row in s.query(RateLimitHit).all()}
    finally:
        s.close()
    assert keys == {"translate:testclient"}
