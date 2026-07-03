"""/klist submissions — public POST (rate-limited, size-capped), bearer-gated GET.

The /klist page is a public static form; filled checklists are personal data, so
they are write-only for visitors and readable only with the LOG_ACCESS_TOKEN
(the /klist/admin page prompts for it).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, Header, HTTPException, Request
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import select
from sqlalchemy.orm import Session

from .. import config
from ..db import get_db
from ..deps import client_ip, require_bearer
from ..models import KlistSubmission
from ..services import rate_limit

router = APIRouter(prefix="/klist", tags=["klist"])

MAX_PAYLOAD_BYTES = 64_000


class SubmissionIn(BaseModel):
    name: str | None = Field(None, max_length=200)
    payload: dict

    @field_validator("payload")
    @classmethod
    def size_cap(cls, v: dict) -> dict:
        if len(json.dumps(v)) > MAX_PAYLOAD_BYTES:
            raise ValueError("payload too large")
        return v


@router.post("/submissions")
def submit(body: SubmissionIn, request: Request, db: Session = Depends(get_db)) -> dict:
    """Store a filled checklist. Public but rate-limited per client IP."""
    limit, window = config.KLIST_RATE
    if not rate_limit.rate_ok(
        db, f"klist:{client_ip(request)}", limit=limit, window=window
    ):
        raise HTTPException(status_code=429, detail="slow down; try again later")
    row = KlistSubmission(
        ts=datetime.now(timezone.utc).replace(tzinfo=None).isoformat(),
        name=body.name,
        payload=body.payload,
        ip=client_ip(request),
    )
    db.add(row)
    db.commit()
    return {"ok": True, "id": row.id}


@router.get("/submissions")
def list_submissions(
    authorization: str | None = Header(None), db: Session = Depends(get_db)
) -> list[dict]:
    """All submissions, newest first. Gated on the shared LOG_ACCESS_TOKEN."""
    require_bearer(authorization)
    rows = db.scalars(select(KlistSubmission).order_by(KlistSubmission.id.desc())).all()
    return [
        {
            "id": r.id,
            "ts": r.ts,
            "name": r.name,
            "payload": r.payload,
            "created_at": r.created_at.isoformat(),
        }
        for r in rows
    ]
