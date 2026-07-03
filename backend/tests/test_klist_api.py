"""Tests for the /klist submissions endpoints (public POST, bearer-gated GET)."""
from fastapi.testclient import TestClient

from backend.app.main import create_app


def _client() -> TestClient:
    return TestClient(create_app())


PAYLOAD = {
    "profile": {"age": "29", "contact": "sig@nal"},
    "features": ["Tall"],
    "ratings": {"General/Blowjobs": {"r": "favourite", "m": "give"}},
    "other_kinks": [],
    "limits": {"hard": "x", "soft": "y"},
}


def test_post_stores_and_returns_id():
    c = _client()
    r = c.post("/klist/submissions", json={"name": "Sam", "payload": PAYLOAD})
    assert r.status_code == 200
    assert r.json()["ok"] is True
    assert isinstance(r.json()["id"], int)


def test_get_requires_bearer():
    c = _client()
    assert c.get("/klist/submissions").status_code == 401
    assert (
        c.get(
            "/klist/submissions", headers={"Authorization": "Bearer wrong"}
        ).status_code
        == 401
    )


def test_get_returns_submissions_newest_first():
    c = _client()
    c.post("/klist/submissions", json={"name": "A", "payload": PAYLOAD})
    c.post("/klist/submissions", json={"name": "B", "payload": PAYLOAD})
    r = c.get("/klist/submissions", headers={"Authorization": "Bearer test-token"})
    assert r.status_code == 200
    rows = r.json()
    assert [x["name"] for x in rows] == ["B", "A"]
    assert rows[0]["payload"]["ratings"]["General/Blowjobs"]["r"] == "favourite"


def test_oversized_payload_rejected():
    c = _client()
    big = {"limits": {"hard": "x" * 70_000}}
    r = c.post("/klist/submissions", json={"name": None, "payload": big})
    assert r.status_code in (400, 422)


def test_malformed_body_rejected():
    c = _client()
    assert (
        c.post("/klist/submissions", json={"payload": "not a dict"}).status_code == 422
    )
