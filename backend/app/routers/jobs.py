"""/jobs interest intake — public POST (rate-limited, size-capped), token-gated GET.

The /jobs board is a public static page; "Pro interest" submissions carry emails,
so they are write-only for visitors and readable only with JOBS_ACCESS_TOKEN
(mirrors the /klist submissions pattern: dedicated secret per surface, per-IP
rate limits on both the writes and the auth attempts).
"""

from __future__ import annotations

import secrets
from datetime import datetime, timezone
from typing import Literal

from fastapi import APIRouter, Depends, Header, HTTPException, Request
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import select
from sqlalchemy.orm import Session

from .. import config
from ..db import get_db
from ..deps import client_ip
from ..models import JobsInterest
from ..services import rate_limit

router = APIRouter(prefix="/jobs", tags=["jobs"])


class InterestIn(BaseModel):
    email: str = Field(max_length=320)
    name: str | None = Field(None, max_length=200)
    tier: Literal["pro", "dossier"]
    target_roles: str | None = Field(None, max_length=2000)
    notes: str | None = Field(None, max_length=2000)

    @field_validator("email", "name", "target_roles", "notes")
    @classmethod
    def strip_ws(cls, v: str | None) -> str | None:
        return v.strip() if isinstance(v, str) else v

    @field_validator("email")
    @classmethod
    def loose_email(cls, v: str) -> str:
        # deliberately loose: just "@" and "." — the point is catching typos/garbage,
        # not RFC 5322 compliance
        if "@" not in v or "." not in v:
            raise ValueError("email must contain '@' and '.'")
        return v


@router.post("/interest")
def submit_interest(
    body: InterestIn, request: Request, db: Session = Depends(get_db)
) -> dict:
    """Store a Pro/dossier interest submission. Public but rate-limited per client IP."""
    limit, window = config.JOBS_INTEREST_RATE
    if not rate_limit.rate_ok(
        db, f"jobs:{client_ip(request)}", limit=limit, window=window
    ):
        raise HTTPException(status_code=429, detail="slow down; try again later")
    row = JobsInterest(
        ts=datetime.now(timezone.utc).replace(tzinfo=None).isoformat(),
        email=body.email,
        name=body.name,
        tier=body.tier,
        target_roles=body.target_roles,
        notes=body.notes,
        ip=client_ip(request),
    )
    db.add(row)
    db.commit()
    return {"ok": True, "id": row.id}


def require_jobs_token(
    authorization: str | None, request: Request, db: Session
) -> None:
    """Gate on JOBS_ACCESS_TOKEN via `Authorization: Bearer <token>` only —
    query-string tokens would end up in access logs. Attempts are
    rate-limited per IP. 500 if unset (misconfig)."""
    limit, window = config.JOBS_ADMIN_RATE
    if not rate_limit.rate_ok(
        db, f"jobs-admin:{client_ip(request)}", limit=limit, window=window
    ):
        raise HTTPException(status_code=429, detail="slow down; try again later")
    if config.JOBS_ACCESS_TOKEN is None:
        raise HTTPException(status_code=500)
    supplied = None
    if authorization and authorization.startswith("Bearer "):
        supplied = authorization.removeprefix("Bearer ").strip()
    if not supplied:
        raise HTTPException(status_code=401)
    # [9] bytes comparison — str compare_digest raises TypeError on non-ASCII
    if not secrets.compare_digest(
        supplied.encode(), config.JOBS_ACCESS_TOKEN.encode()
    ):
        raise HTTPException(status_code=401)


@router.get("/interest")
def list_interest(
    request: Request,
    authorization: str | None = Header(None),
    db: Session = Depends(get_db),
) -> list[dict]:
    """All interest submissions, newest first. Gated on JOBS_ACCESS_TOKEN."""
    require_jobs_token(authorization, request, db)
    rows = db.scalars(select(JobsInterest).order_by(JobsInterest.id.desc())).all()
    return [
        {
            "id": r.id,
            "ts": r.ts,
            "email": r.email,
            "name": r.name,
            "tier": r.tier,
            "target_roles": r.target_roles,
            "notes": r.notes,
            "created_at": r.created_at.isoformat(),
        }
        for r in rows
    ]
