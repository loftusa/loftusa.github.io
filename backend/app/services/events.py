"""Crowd-edit event store (SQLite) — the durable replacement for the JSONL event logs.

Both families (co-authorship graph corrections, affiliation self-service edits) are structurally
identical append-only logs that the pure fold_* functions replay into overlays. This module owns
storage; validation (allowed types, payload keys, per-type byte caps) stays in the routers, exactly
as the old chat_api.py split it. The overlay is folded straight through the real fold_* functions
so it can never diverge from what the nightly baker consumes.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

from experiments.coauthorship.affiliation_events import fold_aff_events
from experiments.coauthorship.overrides import fold_events
from fastapi import HTTPException
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from .. import config
from ..models import AffiliationEvent, CoauthorshipEvent

# family -> (ORM model, fold function, dual-write JSONL path)
_FAMILY = {
    "coauthorship": (CoauthorshipEvent, fold_events, config.CORRECTIONS_PATH),
    "affiliation": (AffiliationEvent, fold_aff_events, config.AFF_EVENTS_PATH),
}


def _family(family: str):
    try:
        return _FAMILY[family]
    except KeyError:
        raise ValueError(f"unknown event family: {family!r}")


def _unique_ts(session: Session, model) -> str:
    """A microsecond-ISO timestamp not already used in this family. Collisions are near-impossible
    but must never silently drop an event (the ts is the unique key), so bump µs until free.
    """
    base = datetime.now(timezone.utc).replace(
        tzinfo=None
    )  # naive UTC == old utcnow() string form
    ts = base.isoformat()
    bump = 0
    while session.scalar(select(func.count()).select_from(model).where(model.ts == ts)):
        bump += 1
        ts = (base + timedelta(microseconds=bump)).isoformat()
    return ts


def append_event(
    session: Session,
    *,
    family: str,
    type: str,
    payload: dict,
    editor: str | None = None,
    note: str | None = None,
    ip: str | None = None,
    max_bytes: int | None = None,
) -> dict:
    """Append one validated event; returns the stored canonical event dict.

    Raises 413 if the event JSON exceeds `max_bytes`, 507 if the family log is full — matching the
    old JSONL guards. Caller is responsible for type/payload validation before calling.
    """
    model, _, _ = _family(family)
    max_bytes = max_bytes or config.MAX_CORRECTION_BYTES
    editor = (editor or "")[:120] or None
    note = (note or "")[:500] or None

    ts = _unique_ts(session, model)
    event = {
        "type": type,
        "payload": payload,
        "editor": editor,
        "note": note,
        "ts": ts,
        "ip": ip,
    }
    if len(json.dumps(event).encode()) > max_bytes:
        raise HTTPException(status_code=413, detail="event too large")
    if (
        session.scalar(select(func.count()).select_from(model))
        >= config.MAX_EVENTS_PER_FAMILY
    ):
        raise HTTPException(status_code=507, detail="event log full")

    session.add(
        model(ts=ts, type=type, payload=payload, editor=editor, note=note, ip=ip)
    )
    session.commit()

    if config.DUAL_WRITE_JSONL:
        _mirror_jsonl(family, event)
    return event


def read_events(session: Session, family: str) -> list[dict]:
    """All events for a family as canonical dicts (insertion order)."""
    model, _, _ = _family(family)
    rows = session.scalars(select(model).order_by(model.id)).all()
    return [r.to_event() for r in rows]


def overlay(session: Session, family: str) -> dict:
    """The folded overlay — identical JSON to the old GET …/overlay."""
    _, fold, _ = _family(family)
    return fold(read_events(session, family))


def delete_by_ts(session: Session, family: str, ts: str) -> dict:
    """Durable admin revert: drop the event(s) with this exact ts. Returns counts; the router maps
    removed==0 to a 404 (preserving the old endpoint's contract)."""
    model, _, _ = _family(family)
    rows = session.scalars(select(model).where(model.ts == ts)).all()
    for r in rows:
        session.delete(r)
    session.commit()
    remaining = session.scalar(select(func.count()).select_from(model))
    return {"ok": True, "removed": len(rows), "remaining": remaining}


def raw_jsonl(session: Session, family: str) -> str:
    """The audit export: one JSON object per line, same shape as the old JSONL file (parsed by
    merge_corrections.py --raw)."""
    return "".join(json.dumps(e) + "\n" for e in read_events(session, family))


def _mirror_jsonl(family: str, event: dict) -> None:
    """Cold-backup dual-write during the cutover window (config.DUAL_WRITE_JSONL)."""
    _, _, path = _family(family)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event) + "\n")
