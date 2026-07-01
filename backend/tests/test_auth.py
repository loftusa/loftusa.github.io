"""P3 auth: JWT mint/verify, idempotent upsert, protected /me, session-version hard-revoke,
expiry, and the internal-key gate. Uses a local app that includes just the auth router.
"""
from backend.app import config
from backend.app.routers.auth import router as auth_router
from backend.app.services import auth as auth_service
from fastapi import FastAPI
from fastapi.testclient import TestClient

app = FastAPI()
app.include_router(auth_router)
client = TestClient(app)
INTERNAL = {"X-Internal-Key": config.INTERNAL_API_KEY}


def bearer(tok: str) -> dict:
    return {"Authorization": f"Bearer {tok}"}


def test_mint_verify_roundtrip(session):
    u = auth_service.upsert_user(session, email="a@b.com", name="A", provider="github")
    claims = auth_service.decode_api_jwt(
        auth_service.mint_api_jwt(u.id, u.email, u.session_version)
    )
    assert claims["sub"] == str(u.id) and claims["ver"] == u.session_version


def test_upsert_is_idempotent_and_normalizes_email(session):
    u1 = auth_service.upsert_user(session, email="X@B.com", provider="github")
    u2 = auth_service.upsert_user(
        session, email="x@b.com", name="X"
    )  # same email, different case
    assert u1.id == u2.id and u2.name == "X"


def test_internal_upsert_requires_key():
    assert (
        client.post("/internal/users/upsert", json={"email": "a@b.com"}).status_code
        == 401
    )
    r = client.post(
        "/internal/users/upsert",
        json={"email": "a@b.com", "provider": "github"},
        headers=INTERNAL,
    )
    assert r.status_code == 200 and r.json()["user_id"]


def test_me_requires_valid_bearer(session):
    u = auth_service.upsert_user(
        session, email="me@b.com", name="Me", provider="github"
    )
    tok = auth_service.mint_api_jwt(u.id, u.email, u.session_version)
    assert client.get("/me").status_code == 401
    assert client.get("/me", headers=bearer("garbage")).status_code == 401
    r = client.get("/me", headers=bearer(tok))
    assert r.status_code == 200 and r.json()["email"] == "me@b.com"


def test_bumping_session_version_revokes_token(session):
    u = auth_service.upsert_user(session, email="r@b.com", provider="github")
    tok = auth_service.mint_api_jwt(u.id, u.email, u.session_version)
    u.session_version += 1  # hard revoke
    session.commit()
    assert client.get("/me", headers=bearer(tok)).status_code == 401


def test_expired_token_rejected(session):
    u = auth_service.upsert_user(session, email="e@b.com", provider="github")
    tok = auth_service.mint_api_jwt(
        u.id, u.email, u.session_version, ttl_s=-10
    )  # already expired
    assert client.get("/me", headers=bearer(tok)).status_code == 401
