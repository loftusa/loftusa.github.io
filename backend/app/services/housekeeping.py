"""Housekeeping service — scheduled cleanup tasks run against the DB.

`run_once` is the entry point called by the background scheduler: prune expired
rate-limit rows and evict conversations that have grown past the capacity cap.
All functions are pure over a Session (injectable `now` + limits for tests).
"""
from __future__ import annotations

import time

from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from .. import config
from ..models import ChatConversation, ChatMessage, RateLimitHit


def prune_rate_limits(
    session: Session,
    now: float | None = None,
    retention_s: float = config.RATE_LIMIT_RETENTION_SECONDS,
) -> int:
    """Delete RateLimitHit rows whose ts is older than `retention_s` seconds. Returns count deleted."""
    now = time.time() if now is None else now
    cutoff = now - retention_s
    result = session.execute(delete(RateLimitHit).where(RateLimitHit.ts < cutoff))
    session.commit()
    return result.rowcount


def evict_stale_conversations(
    session: Session,
    max_conversations: int = config.MAX_CONVERSATIONS,
) -> int:
    """If more than max_conversations rows exist, delete the oldest by updated_at
    (plus their ChatMessage rows) until exactly max_conversations remain.
    Returns the count of evicted ChatConversation rows.

    Note: autoflush=False on the session — we call session.flush() after the ORM
    deletes so the identity map reflects the removals before commit.
    """
    current = session.scalar(select(func.count()).select_from(ChatConversation))
    if current <= max_conversations:
        return 0

    to_evict = current - max_conversations
    oldest = session.scalars(
        select(ChatConversation)
        .order_by(ChatConversation.updated_at.asc())
        .limit(to_evict)
    ).all()

    if not oldest:
        return 0

    user_ids = [c.user_id for c in oldest]
    # Delete messages first (mirrors _delete_with_messages in the conversations service).
    session.execute(delete(ChatMessage).where(ChatMessage.user_id.in_(user_ids)))
    for convo in oldest:
        session.delete(convo)
    # Flush before commit: autoflush=False means ORM deletes aren't emitted until we ask;
    # flushing makes the deletes visible to any subsequent count()/get() in this session.
    session.flush()
    session.commit()
    return len(oldest)


def total_conversation_cost(session: Session) -> float:
    """Return the sum of ChatConversation.cost_usd across all rows.

    This is a cumulative lifetime proxy for spend, not a per-day figure. The chat
    endpoint accumulates cost_usd per conversation via conversations.add_cost(); summing
    here gives a rough total. Returns 0.0 for an empty table.
    """
    result = session.scalar(select(func.sum(ChatConversation.cost_usd)))
    return float(result) if result is not None else 0.0


def run_once(session: Session, now: float | None = None) -> dict:
    """Run all housekeeping tasks in one pass.

    Returns:
        {"pruned_rate_limits": <int>, "evicted_conversations": <int>}
    """
    pruned = prune_rate_limits(session, now=now)
    evicted = evict_stale_conversations(session)
    return {"pruned_rate_limits": pruned, "evicted_conversations": evicted}
