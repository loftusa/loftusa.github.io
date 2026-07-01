"""Idempotent JSONL → SQLite backfill (the dual-write cutover tool).

Run inside the container after deploying the SQLite version:

    python -m backend.app.scripts.import_jsonl

Events use INSERT-OR-IGNORE on the unique `ts`, so this is safe to run repeatedly while the app
dual-writes. Chat logs lack a natural key, so they import once (guarded by migration_state).
"""
from __future__ import annotations

import json
from pathlib import Path

from sqlalchemy import func, select
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.orm import Session

from .. import config
from ..models import AffiliationEvent, ChatLog, CoauthorshipEvent, MigrationState

_FAMILY_MODEL = {"coauthorship": CoauthorshipEvent, "affiliation": AffiliationEvent}


def _read_jsonl(path) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    return [json.loads(ln) for ln in p.read_text().splitlines() if ln.strip()]


def import_events(session: Session, family: str, path) -> dict:
    """Backfill one event family. INSERT-OR-IGNORE on ts → idempotent + duplicate-ts-safe."""
    model = _FAMILY_MODEL[family]
    rows = _read_jsonl(path)
    before = session.scalar(select(func.count()).select_from(model))
    for r in rows:
        session.execute(
            sqlite_insert(model)
            .values(
                ts=r["ts"],
                type=r["type"],
                payload=r.get("payload", {}),
                editor=r.get("editor"),
                note=r.get("note"),
                ip=r.get("ip"),
            )
            .on_conflict_do_nothing(index_elements=["ts"])
        )
    session.commit()
    after = session.scalar(select(func.count()).select_from(model))
    inserted = after - before
    return {"read": len(rows), "inserted": inserted, "skipped": len(rows) - inserted}


def import_chat_logs(session: Session, path) -> dict:
    """One-shot chat-log import (no natural key); a migration_state flag prevents double import."""
    if session.get(MigrationState, "chat_logs_imported") is not None:
        return {"read": 0, "inserted": 0, "skipped": 0}
    rows = _read_jsonl(path)
    for r in rows:
        session.add(
            ChatLog(
                user_id=r.get("user_id"),
                user_message=r.get("user_message", ""),
                bot_response=r.get("bot_response", ""),
                token_latency_ms=r.get("token_latency_ms", -1),
                ts=r.get("timestamp", ""),
                message_count=r.get("message_count", 0),
                token_count=r.get("token_count", 0),
                conversation_cost_usd=r.get("conversation_cost_usd", 0.0),
                rag_chunks_count=r.get("rag_chunks_count", 0),
                rag_sources=r.get("rag_sources", []),
            )
        )
    session.add(MigrationState(key="chat_logs_imported", value=str(len(rows))))
    session.commit()
    return {"read": len(rows), "inserted": len(rows), "skipped": 0}


def main() -> None:
    from ..db import SessionLocal, init_db

    init_db()
    session = SessionLocal()
    try:
        print(
            "coauthorship:",
            import_events(session, "coauthorship", config.CORRECTIONS_PATH),
        )
        print(
            "affiliation:",
            import_events(session, "affiliation", config.AFF_EVENTS_PATH),
        )
        print("chat_logs:", import_chat_logs(session, config.LOG_PATH))
    finally:
        session.close()


if __name__ == "__main__":
    main()
