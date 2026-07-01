"""Shared FastAPI dependencies / guards."""
from __future__ import annotations

from fastapi import HTTPException

from . import config


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
