"""P3 auth — HS256 bearer JWT + user upsert.

Next.js owns OAuth (GitHub/Google via NextAuth). On sign-in the Next BFF S2S-upserts the user here
(gated by INTERNAL_API_KEY) to get a canonical user_id, then mints short-lived HS256 JWTs (shared
API_JWT_SECRET) that the browser sends as `Authorization: Bearer` to this API. We verify the JWT and
the user's `session_version` — bumping session_version hard-revokes every outstanding token.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import jwt
from sqlalchemy import select
from sqlalchemy.orm import Session

from .. import config
from ..models import User


def _utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def mint_api_jwt(
    user_id: int, email: str, session_version: int, ttl_s: int | None = None
) -> str:
    """Mint a short-lived HS256 JWT for the API (used by the Next BFF; also by tests)."""
    now = datetime.now(timezone.utc)
    ttl = config.JWT_TTL_SECONDS if ttl_s is None else ttl_s
    payload = {
        "sub": str(user_id),
        "email": email,
        "ver": session_version,
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(seconds=ttl)).timestamp()),
    }
    return jwt.encode(payload, config.API_JWT_SECRET, algorithm=config.JWT_ALGORITHM)


def decode_api_jwt(token: str) -> dict:
    """Verify + decode; raises jwt exceptions on bad signature / expiry."""
    return jwt.decode(token, config.API_JWT_SECRET, algorithms=[config.JWT_ALGORITHM])


def upsert_user(
    session: Session,
    *,
    email: str,
    name: str | None = None,
    provider: str | None = None,
    provider_sub: str | None = None,
) -> User:
    """Find-or-create by (normalized) email; update mutable fields. Idempotent."""
    email = email.strip().lower()
    user = session.scalar(select(User).where(User.email == email))
    if user is None:
        user = User(
            email=email, name=name, provider=provider, provider_sub=provider_sub
        )
        session.add(user)
    else:
        if name is not None:
            user.name = name
        if provider is not None:
            user.provider = provider
        if provider_sub is not None:
            user.provider_sub = provider_sub
        user.updated_at = _utcnow()
    session.commit()
    return user


def get_user(session: Session, user_id) -> User | None:
    try:
        return session.get(User, int(user_id))
    except (TypeError, ValueError):
        return None
