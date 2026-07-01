"""Tests for the /ws/maps WebSocket presence endpoint.

We build a local FastAPI app that includes only the ws router and drive it with
Starlette's TestClient, which supports websocket_connect() natively.

Two-connections note: each websocket_connect() runs the endpoint in its own thread
with its own event loop. The broadcast from ws2's connect tries to reach ws1 across
event loops; if that send fails, ws1 is silently dropped from manager._active but ws2
still receives its message with count=2 (the count is captured *before* broadcast
iterates). So we assert on the message payload, not on manager.count after the fact.
"""
from __future__ import annotations

import pytest
from backend.app.routers.ws import manager, router
from fastapi import FastAPI
from starlette.testclient import TestClient


@pytest.fixture(autouse=True)
def reset_manager():
    """Ensure a clean manager state for every test in this module."""
    manager._active.clear()
    yield
    manager._active.clear()


def _make_app() -> FastAPI:
    app = FastAPI()
    app.include_router(router)
    return app


# ── tests ─────────────────────────────────────────────────────────────────────


def test_single_connection_receives_presence():
    client = TestClient(_make_app())
    with client.websocket_connect("/ws/maps") as ws:
        msg = ws.receive_json()
        assert msg["type"] == "presence"
        assert msg["count"] >= 1


def test_two_connections_raise_presence_count():
    """The second connection's initial broadcast message must report count == 2."""
    client = TestClient(_make_app())
    with client.websocket_connect("/ws/maps") as ws1:
        msg1 = ws1.receive_json()
        assert msg1["type"] == "presence"
        assert msg1["count"] == 1

        with client.websocket_connect("/ws/maps") as ws2:
            # manager.count is evaluated *before* broadcast iterates, so the message
            # carries count=2 even if the cross-loop send to ws1 later fails.
            msg2 = ws2.receive_json()
            assert msg2["type"] == "presence"
            assert msg2["count"] == 2
