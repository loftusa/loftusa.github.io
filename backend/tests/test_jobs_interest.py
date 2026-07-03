"""Tests for the /jobs interest endpoints (public POST, token-gated GET)."""
from backend.app.main import create_app
from fastapi.testclient import TestClient


def _client() -> TestClient:
    return TestClient(create_app())


VALID = {"email": "jane@example.com", "tier": "pro"}
AUTH = {"Authorization": "Bearer test-jobs-token"}


def test_post_stores_and_returns_ok():
    c = _client()
    r = c.post("/jobs/interest", json=VALID)
    assert r.status_code == 200
    assert r.json()["ok"] is True
    assert isinstance(r.json()["id"], int)


def test_post_full_payload_strips_whitespace():
    c = _client()
    r = c.post(
        "/jobs/interest",
        json={
            "email": "  sam@example.org  ",
            "name": " Sam ",
            "tier": "dossier",
            "target_roles": "  TPM, red-team lead  ",
            "notes": " call me ",
        },
    )
    assert r.status_code == 200
    rows = c.get("/jobs/interest", headers=AUTH).json()
    assert rows[0]["email"] == "sam@example.org"
    assert rows[0]["name"] == "Sam"
    assert rows[0]["tier"] == "dossier"
    assert rows[0]["target_roles"] == "TPM, red-team lead"
    assert rows[0]["notes"] == "call me"


def test_missing_email_rejected():
    c = _client()
    assert c.post("/jobs/interest", json={"tier": "pro"}).status_code == 422


def test_garbage_email_rejected():
    c = _client()
    for bad in ("no-at-sign.com", "no-dot@com", "   ", "@"):
        r = c.post("/jobs/interest", json={"email": bad, "tier": "pro"})
        assert r.status_code == 422, bad


def test_bad_tier_rejected():
    c = _client()
    assert (
        c.post("/jobs/interest", json={"email": "a@b.com", "tier": "free"}).status_code
        == 422
    )


def test_oversized_fields_rejected():
    c = _client()
    assert (
        c.post("/jobs/interest", json={**VALID, "notes": "x" * 2001}).status_code == 422
    )
    assert (
        c.post("/jobs/interest", json={**VALID, "target_roles": "x" * 2001}).status_code
        == 422
    )
    assert (
        c.post(
            "/jobs/interest", json={"email": "a" * 320 + "@b.com", "tier": "pro"}
        ).status_code
        == 422
    )


def test_rate_limited(monkeypatch):
    from backend.app import config

    monkeypatch.setattr(config, "JOBS_INTEREST_RATE", (2, 600))
    c = _client()
    assert c.post("/jobs/interest", json=VALID).status_code == 200
    assert c.post("/jobs/interest", json=VALID).status_code == 200
    assert c.post("/jobs/interest", json=VALID).status_code == 429


def test_get_requires_token():
    c = _client()
    assert c.get("/jobs/interest").status_code == 401
    assert (
        c.get("/jobs/interest", headers={"Authorization": "Bearer wrong"}).status_code
        == 401
    )
    assert c.get("/jobs/interest", params={"token": "wrong"}).status_code == 401


def test_get_rejects_other_surfaces_tokens():
    """The chat-log and klist tokens must NOT open the jobs viewer (separate secrets)."""
    c = _client()
    for tok in ("test-token", "test-klist-token"):
        r = c.get("/jobs/interest", headers={"Authorization": f"Bearer {tok}"})
        assert r.status_code == 401, tok


def test_get_returns_submissions_newest_first():
    c = _client()
    c.post("/jobs/interest", json={"email": "a@x.com", "tier": "pro"})
    c.post("/jobs/interest", json={"email": "b@x.com", "tier": "dossier"})
    r = c.get("/jobs/interest", headers=AUTH)
    assert r.status_code == 200
    rows = r.json()
    assert [x["email"] for x in rows] == ["b@x.com", "a@x.com"]
    assert rows[0]["tier"] == "dossier"
    assert rows[0]["ts"] and rows[0]["created_at"]


def test_get_rejects_query_token():
    """Query-string tokens land in access logs — header-only auth."""
    c = _client()
    r = c.get("/jobs/interest", params={"token": "test-jobs-token"})
    assert r.status_code == 401


def test_get_rejects_non_ascii_token():
    """Non-ASCII supplied token must 401, not crash compare_digest."""
    c = _client()
    # httpx refuses non-ascii str headers; send raw latin-1 bytes like a
    # hostile client would (uvicorn decodes header bytes as latin-1).
    r = c.get(
        "/jobs/interest",
        headers={b"Authorization": "Bearer t\u00f6ken".encode("latin-1")},
    )
    assert r.status_code == 401
