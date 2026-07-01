"""Affiliation self-service endpoints — wire-identical to the old chat_api.py routes, backed by
the SQLite event store. Full payload validation lives here so a bad event can't reach the fold;
the roster-collision check reads the folded join set from the DB instead of the JSONL file.
"""
from __future__ import annotations

import json
from typing import Literal

from experiments.coauthorship.affiliation_events import (
    ENTRY_TYPES,
    norm_person,
)
from experiments.coauthorship.affiliation_events import (
    FIELD_CAPS as AFF_FIELD_CAPS,
)
from fastapi import APIRouter, Depends, Header, HTTPException, Request
from fastapi.responses import Response
from pydantic import BaseModel
from sqlalchemy.orm import Session

from .. import config
from ..db import get_db
from ..deps import client_ip, require_bearer
from ..services import events, rate_limit

router = APIRouter(prefix="/affiliations", tags=["affiliations"])

AFF_EVENT_TYPES = {  # event type -> required payload keys
    "aff_entry_set": {"person", "org", "type"},
    "aff_entry_remove": {"person", "org"},
    "aff_city": {"person", "city"},
    "aff_join": {"name"},
    "aff_confirm": {"person"},
}


class AffEvent(BaseModel):
    type: Literal[
        "aff_entry_set", "aff_entry_remove", "aff_city", "aff_join", "aff_confirm"
    ]
    payload: dict
    editor: str | None = None
    note: str | None = None


def _roster_norms(db: Session) -> set[str]:
    """Normalized roster names: seeds shipped in the image + already-folded joins (from the DB)."""
    names: set[str] = set()
    if config.AFF_SEEDS_PATH.exists():
        names = {
            norm_person(s["name"])
            for s in json.loads(config.AFF_SEEDS_PATH.read_text())
        }
    names |= set(events.overlay(db, "affiliation")["join"])
    return names


def _check_aff_fields(payload: dict) -> None:
    for field, cap in AFF_FIELD_CAPS.items():
        v = payload.get(field)
        if v is not None and len(str(v)) > cap:
            raise HTTPException(
                status_code=422, detail=f"`{field}` longer than {cap} chars"
            )


def _check_aff_entry(spec: dict) -> None:
    if not isinstance(spec, dict) or not str(spec.get("org", "")).strip():
        raise HTTPException(status_code=422, detail="entry needs a non-empty `org`")
    if spec.get("type") not in ENTRY_TYPES:
        raise HTTPException(
            status_code=422, detail=f"entry `type` must be one of {sorted(ENTRY_TYPES)}"
        )
    _check_aff_fields(spec)


@router.post("/corrections")
def submit_aff_event(
    event: AffEvent, request: Request, db: Session = Depends(get_db)
) -> dict:
    """Append one affiliation self-service event. Open, guarded by the same rate limit + caps;
    full validation so a bad event can't reach the fold."""
    ip = client_ip(request)
    limit, window = config.CORRECTION_RATE
    if not rate_limit.rate_ok(db, ip, limit=limit, window=window):
        raise HTTPException(status_code=429, detail="too many edits; try again later")

    p = event.payload
    missing = AFF_EVENT_TYPES[event.type] - {k for k, v in p.items() if v is not None}
    if missing:
        raise HTTPException(
            status_code=422, detail=f"payload missing keys: {sorted(missing)}"
        )
    if event.type == "aff_entry_set":
        _check_aff_entry(p)
    else:
        _check_aff_fields(p)
    if event.type == "aff_join":
        entries = p.get("entries") or []
        if not isinstance(entries, list) or len(entries) > 10:
            raise HTTPException(
                status_code=422, detail="`entries` must be a list of at most 10"
            )
        for spec in entries:
            _check_aff_entry(spec)
        if norm_person(p["name"]) in _roster_norms(db):
            raise HTTPException(
                status_code=409,
                detail="that name is already on the map — edit the row instead",
            )

    cap = (
        config.MAX_JOIN_EVENT_BYTES
        if event.type == "aff_join"
        else config.MAX_CORRECTION_BYTES
    )
    events.append_event(
        db,
        family="affiliation",
        type=event.type,
        payload=p,
        editor=event.editor,
        note=event.note,
        ip=ip,
        max_bytes=cap,
    )
    return {"ok": True}


@router.get("/overlay")
def aff_overlay(db: Session = Depends(get_db)) -> dict:
    """Open, read-only: the folded affiliation overlay (live preview + nightly merge source)."""
    return events.overlay(db, "affiliation")


@router.get("/corrections")
def export_aff_events(
    authorization: str | None = Header(None), db: Session = Depends(get_db)
):
    """Bearer-protected raw event log (audit trail; carries editor + ip)."""
    require_bearer(authorization)
    return Response(
        content=events.raw_jsonl(db, "affiliation"), media_type="text/plain"
    )


@router.delete("/corrections")
def delete_aff_event(
    ts: str, authorization: str | None = Header(None), db: Session = Depends(get_db)
) -> dict:
    """Bearer-protected admin revert by exact ts."""
    require_bearer(authorization)
    res = events.delete_by_ts(db, "affiliation", ts)
    if res["removed"] == 0:
        raise HTTPException(status_code=404, detail=f"no event with ts={ts}")
    return res
