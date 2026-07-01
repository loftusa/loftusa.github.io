"""Durable sliding-window per-IP rate limiter (replaces the in-memory `_correction_hits` dict).

Hits are rows in `rate_limit_hits`, so limits survive a Fly restart/deploy. A rejected request
does not store a hit (matching the old limiter). `now` is injectable for deterministic tests; in
production it defaults to wall-clock epoch seconds.
"""
from __future__ import annotations

import time

from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from ..models import RateLimitHit


def hit_count(
    session: Session, client_ip: str, *, window: float, now: float | None = None
) -> int:
    """Hits for this IP still inside the window."""
    now = time.time() if now is None else now
    cutoff = now - window
    return session.scalar(
        select(func.count())
        .select_from(RateLimitHit)
        .where(RateLimitHit.client_ip == client_ip, RateLimitHit.ts >= cutoff)
    )


def rate_ok(
    session: Session,
    client_ip: str,
    *,
    limit: int,
    window: float,
    now: float | None = None,
) -> bool:
    """True (and records a hit) if this IP is under `limit` within the trailing `window` seconds.
    False (recording nothing) once at/over the limit."""
    now = time.time() if now is None else now
    # prune this IP's expired hits so the table can't grow unbounded
    session.execute(
        delete(RateLimitHit).where(
            RateLimitHit.client_ip == client_ip, RateLimitHit.ts < now - window
        )
    )
    if hit_count(session, client_ip, window=window, now=now) >= limit:
        session.commit()  # persist the prune even on rejection
        return False
    session.add(RateLimitHit(client_ip=client_ip, ts=now))
    session.commit()
    return True
