"""/klist submissions — public POST (rate-limited, size-capped), bearer-gated GET.

The /klist page is a public static form; filled checklists are personal data, so
they are write-only for visitors and readable only with the LOG_ACCESS_TOKEN
(the /klist/admin page prompts for it).
"""

from __future__ import annotations

import json
import secrets
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, Header, HTTPException, Request
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import select
from sqlalchemy.orm import Session

from .. import config
from ..db import get_db
from ..deps import client_ip
from ..models import KlistSchemaItem, KlistSubmission
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


def require_klist_bearer(
    authorization: str | None, request: Request, db: Session
) -> None:
    """Gate on KLIST_ACCESS_TOKEN (the short /klist/admin PIN). Attempts are
    rate-limited per IP because the PIN is short. 500 if unset (misconfig)."""
    limit, window = config.KLIST_ADMIN_RATE
    if not rate_limit.rate_ok(
        db, f"klist-admin:{client_ip(request)}", limit=limit, window=window
    ):
        raise HTTPException(status_code=429, detail="slow down; try again later")
    if config.KLIST_ACCESS_TOKEN is None:
        raise HTTPException(status_code=500)
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401)
    supplied = authorization.removeprefix("Bearer ").strip()
    if not secrets.compare_digest(supplied, config.KLIST_ACCESS_TOKEN):
        raise HTTPException(status_code=401)


@router.get("/submissions")
def list_submissions(
    request: Request,
    authorization: str | None = Header(None),
    db: Session = Depends(get_db),
) -> list[dict]:
    """All submissions, newest first. Gated on the /klist admin PIN."""
    require_klist_bearer(authorization, request, db)
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


class SchemaItemIn(BaseModel):
    section: str = Field(min_length=2, max_length=80)
    item: str = Field(min_length=2, max_length=120)


@router.get("/schema")
def list_schema(db: Session = Depends(get_db)) -> list[dict]:
    """Visitor-added sections/items, oldest first (stable render order)."""
    rows = db.scalars(select(KlistSchemaItem).order_by(KlistSchemaItem.id)).all()
    return [{"id": r.id, "section": r.section, "item": r.item} for r in rows]


@router.post("/schema")
def add_schema_item(
    body: SchemaItemIn, request: Request, db: Session = Depends(get_db)
) -> dict:
    """Permanently add a tile (and implicitly its section) to the checklist.

    Open to anyone with the form link, so: rate-limited per IP and deduped
    case-insensitively. Deletion is admin-only."""
    limit, window = config.KLIST_SCHEMA_RATE
    if not rate_limit.rate_ok(
        db, f"klist-schema:{client_ip(request)}", limit=limit, window=window
    ):
        raise HTTPException(status_code=429, detail="slow down; try again later")
    section, item = body.section.strip(), body.item.strip()
    if len(section) < 2 or len(item) < 2:
        raise HTTPException(status_code=422, detail="section/item too short")
    dupe = [
        r
        for r in db.scalars(select(KlistSchemaItem)).all()
        if r.section.lower() == section.lower() and r.item.lower() == item.lower()
    ]
    if dupe:
        raise HTTPException(status_code=409, detail="already on the list")
    row = KlistSchemaItem(section=section, item=item, ip=client_ip(request))
    db.add(row)
    db.commit()
    return {"ok": True, "id": row.id}


@router.delete("/schema/{item_id}")
def delete_schema_item(
    item_id: int,
    request: Request,
    authorization: str | None = Header(None),
    db: Session = Depends(get_db),
) -> dict:
    """Remove a visitor-added tile (admin PIN only)."""
    require_klist_bearer(authorization, request, db)
    row = db.get(KlistSchemaItem, item_id)
    if row is None:
        raise HTTPException(status_code=404)
    db.delete(row)
    db.commit()
    return {"ok": True}
