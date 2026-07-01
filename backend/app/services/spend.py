"""Durable LLM spend meter — one SpendEvent row per metered turn, summed over a trailing
24h window to enforce config.DAILY_COST_CEILING_USD in /chat *before* the LLM call.

The per-conversation cost cap alone doesn't bound aggregate spend (fresh conversation ids
are free); this table does. `now` is injectable for deterministic tests.
"""
from __future__ import annotations

import time

from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from ..models import SpendEvent


def record(session: Session, usd: float, now: float | None = None) -> None:
    """Append one spend row (commits)."""
    assert usd >= 0, f"negative spend: {usd}"
    session.add(SpendEvent(ts=time.time() if now is None else now, usd=usd))
    session.commit()


def total_last_24h(session: Session, now: float | None = None) -> float:
    now = time.time() if now is None else now
    total = session.scalar(
        select(func.sum(SpendEvent.usd)).where(SpendEvent.ts >= now - 86400)
    )
    return float(total) if total is not None else 0.0


def prune(session: Session, retention_s: float, now: float | None = None) -> int:
    """Delete rows older than retention_s seconds; returns count deleted (commits)."""
    now = time.time() if now is None else now
    result = session.execute(
        delete(SpendEvent).where(SpendEvent.ts < now - retention_s)
    )
    session.commit()
    return result.rowcount
