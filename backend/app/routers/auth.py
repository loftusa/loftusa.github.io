"""P3 auth endpoints: the S2S user-upsert (NextAuth sign-in -> here) and a protected /me.

Auth is optional site-wide; chat + the maps stay anonymous. Only account endpoints depend on
`get_current_user`.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session

from ..db import get_db
from ..deps import get_current_user, require_internal_key
from ..models import User
from ..services import auth as auth_service

router = APIRouter(tags=["auth"])


class UpsertRequest(BaseModel):
    email: str
    name: str | None = None
    provider: str | None = None
    provider_sub: str | None = None


@router.post("/internal/users/upsert")
def upsert(
    req: UpsertRequest,
    db: Session = Depends(get_db),
    _: None = Depends(require_internal_key),
) -> dict:
    """Called by the NextAuth signIn callback (INTERNAL_API_KEY-gated) to get a canonical user_id."""
    user = auth_service.upsert_user(
        db,
        email=req.email,
        name=req.name,
        provider=req.provider,
        provider_sub=req.provider_sub,
    )
    return {
        "user_id": user.id,
        "email": user.email,
        "session_version": user.session_version,
    }


@router.get("/me")
def me(user: User = Depends(get_current_user)) -> dict:
    """The signed-in user (proves the bearer-JWT round-trip)."""
    return {
        "id": user.id,
        "email": user.email,
        "name": user.name,
        "provider": user.provider,
        "role": user.role,
    }
