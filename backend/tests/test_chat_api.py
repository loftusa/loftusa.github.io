"""Chat endpoint tests — only the LLM-free paths (the streaming path needs cerebras/chromadb,
exercised in deploy). Covers the durable cost gate, /reset, and /health.
"""
from backend.app import config
from backend.app.db import SessionLocal
from backend.app.main import app
from backend.app.services import conversations as conv
from fastapi.testclient import TestClient

client = TestClient(app)


def test_health_ok():
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_reset_returns_ok():
    assert client.post("/reset", json={"user_id": "u"}).status_code == 200


def test_over_budget_conversation_returns_429_before_llm():
    # seed a conversation already at its cost cap; /chat must 429 before ever calling the LLM
    s = SessionLocal()
    conv.get_or_create(s, "broke")
    conv.add_cost(s, "broke", config.MAX_COST_PER_CONVERSATION)
    s.close()
    r = client.post("/chat", json={"user_id": "broke", "message": "hi"})
    assert r.status_code == 429


def test_reset_clears_history():
    s = SessionLocal()
    conv.get_or_create(s, "u2")
    conv.append_message(s, "u2", "user", "hi")
    s.close()
    client.post("/reset", json={"user_id": "u2"})
    s2 = SessionLocal()
    try:
        assert conv.messages(s2, "u2") == []
    finally:
        s2.close()
