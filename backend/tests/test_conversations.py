"""Durable conversation store: per-conversation cost cap that survives restart, persisted
message history, LRU eviction, and turn trimming — the DB versions of chat_api.py's in-memory
`conversations` OrderedDict + Conversation dataclass.
"""
import datetime as dt

from backend.app.db import SessionLocal
from backend.app.services import conversations as conv


def test_generates_id_when_none(session):
    uid = conv.get_or_create(session, None)
    assert uid and isinstance(uid, str)


def test_reuses_existing_id_without_duplicating(session):
    conv.get_or_create(session, "alice")
    assert conv.get_or_create(session, "alice") == "alice"
    assert conv.count(session) == 1


def test_cost_accumulates(session):
    conv.get_or_create(session, "u")
    assert conv.get_cost(session, "u") == 0.0
    conv.add_cost(session, "u", 0.5)
    conv.add_cost(session, "u", 0.25)
    assert conv.get_cost(session, "u") == 0.75


def test_over_budget(session):
    conv.get_or_create(session, "u")
    conv.add_cost(session, "u", 2.0)
    assert conv.over_budget(session, "u", cap=2.0)
    assert not conv.over_budget(session, "unknown", cap=2.0)


def test_cost_survives_a_fresh_session(session):
    s1 = SessionLocal()
    conv.get_or_create(s1, "u")
    conv.add_cost(s1, "u", 1.5)
    s1.close()
    s2 = SessionLocal()
    try:
        assert conv.get_cost(s2, "u") == 1.5  # restart-proof cost cap
    finally:
        s2.close()


def test_messages_roundtrip_in_order(session):
    conv.get_or_create(session, "u")
    conv.append_message(session, "u", "user", "hi")
    conv.append_message(session, "u", "assistant", "hello")
    assert conv.messages(session, "u") == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]


def test_reset_clears_conversation_and_messages(session):
    conv.get_or_create(session, "u")
    conv.append_message(session, "u", "user", "hi")
    conv.add_cost(session, "u", 1.0)
    conv.reset(session, "u")
    assert conv.messages(session, "u") == []
    assert conv.get_cost(session, "u") == 0.0  # fully gone → fresh


def test_trim_keeps_last_n_turns(session):
    conv.get_or_create(session, "u")
    for i in range(5):
        conv.append_message(session, "u", "user", f"u{i}")
        conv.append_message(session, "u", "assistant", f"a{i}")
    conv.trim_turns(
        session, "u", max_turns=2
    )  # keep last 2 user+assistant pairs = 4 messages
    msgs = conv.messages(session, "u")
    assert [m["content"] for m in msgs] == ["u3", "a3", "u4", "a4"]


def test_lru_evicts_least_recently_used(session):
    t = dt.datetime(2026, 1, 1)
    conv.get_or_create(session, "a", max_conversations=2, now=t)
    conv.get_or_create(
        session, "b", max_conversations=2, now=t + dt.timedelta(seconds=1)
    )
    conv.get_or_create(
        session, "a", max_conversations=2, now=t + dt.timedelta(seconds=2)
    )  # touch a
    conv.get_or_create(
        session, "c", max_conversations=2, now=t + dt.timedelta(seconds=3)
    )  # evict b
    assert conv.exists(session, "a") and conv.exists(session, "c")
    assert not conv.exists(session, "b")
