"""Spend meter + the daily cost ceiling gate on /chat.

DAILY_COST_CEILING_USD used to be dead config — defined but enforced nowhere, so aggregate
spend across many fresh conversations was unbounded despite the per-conversation cap.
"""
from __future__ import annotations

from backend.app import config
from backend.app.db import SessionLocal
from backend.app.main import app
from backend.app.services import spend
from fastapi.testclient import TestClient

client = TestClient(app)


def test_total_only_counts_trailing_24h(session):
    t0 = 1_000_000.0
    spend.record(session, 5.0, now=t0)
    spend.record(session, 3.0, now=t0 + 90_000)  # 25h later — first row aged out
    assert spend.total_last_24h(session, now=t0 + 90_000) == 3.0


def test_prune_removes_only_expired_rows(session):
    spend.record(session, 1.0, now=0.0)
    spend.record(session, 1.0, now=1000.0)
    assert spend.prune(session, retention_s=500, now=1000.0) == 1
    assert spend.total_last_24h(session, now=1000.0) == 1.0


def test_chat_429_when_daily_ceiling_hit():
    s = SessionLocal()
    try:
        spend.record(s, config.DAILY_COST_CEILING_USD)
    finally:
        s.close()
    r = client.post("/chat", json={"user_id": "fresh-user", "message": "hi"})
    assert r.status_code == 429
    assert "budget" in r.json()["detail"].lower()
