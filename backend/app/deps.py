"""Shared FastAPI dependencies / guards."""
from __future__ import annotations

from fastapi import Depends, Header, HTTPException
from sqlalchemy.orm import Session

from . import config
from .db import get_db
from .models import User
from .services import auth as auth_service


def require_bearer(authorization: str | None) -> None:
    """Gate an endpoint on the shared LOG_ACCESS_TOKEN. 500 if the token is unset (misconfig),
    401 if the header is missing/malformed/wrong. Same contract as the old chat_api.require_bearer.
    """
    if config.LOG_ACCESS_TOKEN is None:
        raise HTTPException(status_code=500)
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401)
    if authorization.removeprefix("Bearer ").strip() != config.LOG_ACCESS_TOKEN:
        raise HTTPException(status_code=401)


def client_ip(request) -> str:
    return request.client.host if request.client else "unknown"


def require_internal_key(x_internal_key: str | None = Header(None)) -> None:
    """Gate the S2S internal endpoints (NextAuth -> FastAPI user upsert) on INTERNAL_API_KEY."""
    if config.INTERNAL_API_KEY is None:
        raise HTTPException(status_code=500)
    if x_internal_key != config.INTERNAL_API_KEY:
        raise HTTPException(status_code=401, detail="bad internal key")


def get_current_user(
    authorization: str | None = Header(None), db: Session = Depends(get_db)
) -> User:
    """Verify the bearer JWT + the user's session_version; return the User. 401 on any failure.
    Auth is OPTIONAL site-wide — only account endpoints depend on this."""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="missing bearer token")
    token = authorization.removeprefix("Bearer ").strip()
    try:
        claims = auth_service.decode_api_jwt(token)
    except Exception:
        raise HTTPException(status_code=401, detail="invalid token")
    user = auth_service.get_user(db, claims.get("sub"))
    if user is None:
        raise HTTPException(status_code=401, detail="unknown user")
    if user.session_version != claims.get("ver"):
        raise HTTPException(status_code=401, detail="session revoked")
    return user
