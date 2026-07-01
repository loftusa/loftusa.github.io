"""Tests for the housekeeping service: prune_rate_limits, evict_stale_conversations,
total_conversation_cost, and run_once."""
from datetime import datetime, timedelta

import pytest
from backend.app.models import ChatConversation, ChatMessage, RateLimitHit
from backend.app.services.housekeeping import (
    evict_stale_conversations,
    prune_rate_limits,
    run_once,
    total_conversation_cost,
)
from sqlalchemy import select

# ── helpers ───────────────────────────────────────────────────────────────────


def _convo(
    user_id: str, updated_at: datetime, cost_usd: float = 0.0
) -> ChatConversation:
    return ChatConversation(
        user_id=user_id,
        cost_usd=cost_usd,
        created_at=updated_at,
        updated_at=updated_at,
    )


# ── prune_rate_limits ─────────────────────────────────────────────────────────


def test_prune_rate_limits_deletes_old_rows(session):
    now = 1_000_000.0
    retention = 600.0

    session.add(RateLimitHit(client_ip="1.2.3.4", ts=now - 700))  # old: pruned
    session.add(RateLimitHit(client_ip="1.2.3.4", ts=now - 100))  # recent: kept
    session.add(RateLimitHit(client_ip="5.6.7.8", ts=now))  # recent: kept
    session.commit()

    deleted = prune_rate_limits(session, now=now, retention_s=retention)

    assert deleted == 1
    remaining = session.scalars(select(RateLimitHit)).all()
    assert len(remaining) == 2


def test_prune_rate_limits_nothing_to_prune(session):
    now = 1_000_000.0
    session.add(RateLimitHit(client_ip="1.2.3.4", ts=now - 10))
    session.commit()

    deleted = prune_rate_limits(session, now=now, retention_s=600.0)

    assert deleted == 0
    assert len(session.scalars(select(RateLimitHit)).all()) == 1


def test_prune_rate_limits_empty_table(session):
    assert prune_rate_limits(session, now=1_000_000.0, retention_s=600.0) == 0


# ── evict_stale_conversations ─────────────────────────────────────────────────


def test_evict_stale_conversations_removes_oldest(session):
    base = datetime(2026, 1, 1)
    for i in range(5):
        session.add(_convo(f"user_{i}", base + timedelta(hours=i)))
    session.commit()

    evicted = evict_stale_conversations(session, max_conversations=3)

    assert evicted == 2
    remaining = session.scalars(select(ChatConversation)).all()
    assert len(remaining) == 3
    remaining_ids = {c.user_id for c in remaining}
    assert "user_0" not in remaining_ids  # oldest — gone
    assert "user_1" not in remaining_ids  # second oldest — gone
    assert {"user_2", "user_3", "user_4"} == remaining_ids


def test_evict_stale_conversations_also_removes_messages(session):
    base = datetime(2026, 1, 1)
    for i in range(4):
        session.add(_convo(f"user_{i}", base + timedelta(hours=i)))
    session.add(ChatMessage(user_id="user_0", role="user", content="hi"))
    session.add(ChatMessage(user_id="user_0", role="assistant", content="hello"))
    session.commit()

    evict_stale_conversations(session, max_conversations=3)

    orphaned = session.scalars(
        select(ChatMessage).where(ChatMessage.user_id == "user_0")
    ).all()
    assert orphaned == []


def test_evict_stale_conversations_no_op_under_cap(session):
    base = datetime(2026, 1, 1)
    for i in range(2):
        session.add(_convo(f"user_{i}", base + timedelta(hours=i)))
    session.commit()

    evicted = evict_stale_conversations(session, max_conversations=5)

    assert evicted == 0
    assert len(session.scalars(select(ChatConversation)).all()) == 2


# ── total_conversation_cost ───────────────────────────────────────────────────


def test_total_conversation_cost_sums_all_rows(session):
    base = datetime(2026, 1, 1)
    session.add(_convo("a", base, cost_usd=1.0))
    session.add(_convo("b", base, cost_usd=2.5))
    session.commit()

    assert total_conversation_cost(session) == pytest.approx(3.5)


def test_total_conversation_cost_empty_table(session):
    assert total_conversation_cost(session) == 0.0


# ── run_once ──────────────────────────────────────────────────────────────────


def test_run_once_returns_summary(session):
    now = 1_000_000.0
    # ~8 days old, older than RATE_LIMIT_RETENTION_SECONDS (7 days = 604 800 s)
    session.add(RateLimitHit(client_ip="x", ts=now - 700_000))
    base = datetime(2026, 1, 1)
    for i in range(3):
        session.add(_convo(f"user_{i}", base + timedelta(hours=i)))
    session.commit()

    result = run_once(session, now=now)

    assert result["pruned_rate_limits"] == 1
    assert result["evicted_conversations"] == 0  # 3 <= default MAX_CONVERSATIONS (1000)
