"""Co-authorship graph crowd-edit endpoints — wire-identical to the old chat_api.py routes, now
backed by the SQLite event store. Validation (allowed types, required payload keys, the `between`
shape) stays here; storage + folding live in services.events.
"""
from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, Depends, Header, HTTPException, Request
from fastapi.responses import Response
from pydantic import BaseModel
from sqlalchemy.orm import Session

from .. import config
from ..db import get_db
from ..deps import client_ip, require_bearer
from ..services import events, rate_limit

router = APIRouter(prefix="/coauthorship", tags=["coauthorship"])

CORRECTION_TYPES = {  # event type -> required payload keys
    "node_label": {"id", "label"},
    "node_community": {"id", "community"},
    "node_url": {"id"},
    "node_photo": {"id", "filename"},
    "remove_node": {"id"},
    "paper_rename": {"old", "new"},
    "remove_paper": {"between", "title"},
    "add_paper": {"between", "title"},
    "remove_edge": {"between"},
}


class Correction(BaseModel):
    type: Literal[
        "node_label",
        "node_community",
        "node_url",
        "node_photo",
        "remove_node",
        "paper_rename",
        "remove_paper",
        "add_paper",
        "remove_edge",
    ]
    payload: dict
    editor: str | None = None
    note: str | None = None


@router.post("/corrections")
def submit_correction(
    correction: Correction, request: Request, db: Session = Depends(get_db)
) -> dict:
    """Append one crowd edit. Open, but guarded by per-IP rate limit + size caps."""
    ip = client_ip(request)
    limit, window = config.CORRECTION_RATE
    if not rate_limit.rate_ok(db, ip, limit=limit, window=window):
        raise HTTPException(
            status_code=429, detail="too many corrections; try again later"
        )

    required = CORRECTION_TYPES[correction.type]
    missing = required - set(correction.payload)
    if missing:
        raise HTTPException(
            status_code=422, detail=f"payload missing keys: {sorted(missing)}"
        )
    if "between" in correction.payload:
        pair = correction.payload["between"]
        if not (isinstance(pair, list) and len(pair) == 2):
            raise HTTPException(
                status_code=422, detail="`between` must be a 2-item list"
            )

    events.append_event(
        db,
        family="coauthorship",
        type=correction.type,
        payload=correction.payload,
        editor=correction.editor,
        note=correction.note,
        ip=ip,
        max_bytes=config.MAX_CORRECTION_BYTES,
    )
    return {"ok": True}


@router.get("/overlay")
def correction_overlay(db: Session = Depends(get_db)) -> dict:
    """Open, read-only: the live merged overlay (pending edits pre-rebuild)."""
    return events.overlay(db, "coauthorship")


@router.get("/corrections")
def export_corrections(
    authorization: str | None = Header(None), db: Session = Depends(get_db)
):
    """Bearer-protected raw event log as JSONL text (parsed by merge_corrections.py --raw)."""
    require_bearer(authorization)
    return Response(
        content=events.raw_jsonl(db, "coauthorship"), media_type="text/plain"
    )


@router.delete("/corrections")
def delete_correction(
    ts: str, authorization: str | None = Header(None), db: Session = Depends(get_db)
) -> dict:
    """Bearer-protected admin revert: drop the event(s) with this exact ts."""
    require_bearer(authorization)
    res = events.delete_by_ts(db, "coauthorship", ts)
    if res["removed"] == 0:
        raise HTTPException(status_code=404, detail=f"no event with ts={ts}")
    return res
