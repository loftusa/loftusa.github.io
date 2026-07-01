"""Durable conversation store — the DB version of chat_api.py's in-memory `conversations`
OrderedDict + Conversation dataclass.

What SQLite buys us: the per-conversation cost cap survives a Fly restart/deploy (it used to reset
to $0 on every redeploy), and message history persists. LRU eviction by `updated_at` bounds the
table just like the old OrderedDict capacity bound. `now` is injectable for deterministic tests.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from .. import config
from ..models import ChatConversation, ChatMessage


def _utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def exists(session: Session, user_id: str) -> bool:
    return session.get(ChatConversation, user_id) is not None


def count(session: Session) -> int:
    return session.scalar(select(func.count()).select_from(ChatConversation))


def get_or_create(
    session: Session,
    user_id: str | None,
    *,
    max_conversations: int | None = None,
    now: datetime | None = None,
) -> str:
    """Return the canonical user_id, creating the conversation row if missing (and touching its
    recency if present). Evicts the least-recently-used row once over capacity — the durable
    analogue of the old OrderedDict.move_to_end / popitem(last=False)."""
    now = now or _utcnow()
    cap = max_conversations or config.MAX_CONVERSATIONS
    if user_id is None:
        user_id = str(uuid.uuid4())

    convo = session.get(ChatConversation, user_id)
    if convo is not None:
        convo.updated_at = now  # move-to-end (most recently used)
        session.commit()
        return user_id

    while count(session) >= cap:  # evict LRU before inserting
        oldest = session.scalar(
            select(ChatConversation)
            .order_by(ChatConversation.updated_at.asc())
            .limit(1)
        )
        if oldest is None:
            break
        _delete_with_messages(session, oldest.user_id)

    session.add(ChatConversation(user_id=user_id, created_at=now, updated_at=now))
    session.commit()
    return user_id


def get_cost(session: Session, user_id: str) -> float:
    convo = session.get(ChatConversation, user_id)
    return convo.cost_usd if convo else 0.0


def add_cost(session: Session, user_id: str, amount: float) -> float:
    convo = session.get(ChatConversation, user_id)
    if convo is None:
        convo = ChatConversation(user_id=user_id)
        session.add(convo)
    convo.cost_usd += amount
    session.commit()
    return convo.cost_usd


def over_budget(session: Session, user_id: str, cap: float) -> bool:
    return get_cost(session, user_id) >= cap


def messages(session: Session, user_id: str) -> list[dict]:
    rows = session.scalars(
        select(ChatMessage)
        .where(ChatMessage.user_id == user_id)
        .order_by(ChatMessage.id)
    ).all()
    return [{"role": r.role, "content": r.content} for r in rows]


def append_message(session: Session, user_id: str, role: str, content: str) -> None:
    session.add(ChatMessage(user_id=user_id, role=role, content=content))
    convo = session.get(ChatConversation, user_id)
    if convo is not None:
        convo.message_count = (convo.message_count or 0) + 1
        convo.updated_at = _utcnow()
    session.commit()


def trim_turns(session: Session, user_id: str, max_turns: int) -> None:
    """Keep only the most recent `max_turns` user+assistant pairs (matches the old
    `while len(messages) >= MAX_TURNS*2: pop(0)` trimming, but durable)."""
    keep = max_turns * 2
    ids = session.scalars(
        select(ChatMessage.id)
        .where(ChatMessage.user_id == user_id)
        .order_by(ChatMessage.id)
    ).all()
    if len(ids) <= keep:
        return
    drop = ids[: len(ids) - keep]
    session.execute(delete(ChatMessage).where(ChatMessage.id.in_(drop)))
    session.commit()


def reset(session: Session, user_id: str) -> None:
    """Clear a conversation entirely (history + cost) — the durable `/reset`."""
    _delete_with_messages(session, user_id)
    session.commit()


def _delete_with_messages(session: Session, user_id: str) -> None:
    session.execute(delete(ChatMessage).where(ChatMessage.user_id == user_id))
    convo = session.get(ChatConversation, user_id)
    if convo is not None:
        session.delete(convo)
    # flush now (autoflush is off): emits the DELETE so the eviction loop's count() drops, and
    # evicts the instance from the identity map so a later get() returns None, not a stale row.
    session.flush()
