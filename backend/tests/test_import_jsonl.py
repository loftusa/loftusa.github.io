"""The JSONL→SQLite backfill must be idempotent (safe to re-run during the dual-write cutover)
and must preserve `ts` exactly so the folded overlay is unchanged.
"""
import json

from backend.app.scripts import import_jsonl
from backend.app.services import events as ev
from experiments.coauthorship.overrides import fold_events


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_import_events_preserves_overlay_and_is_idempotent(session, tmp_path):
    rows = [
        {
            "type": "node_label",
            "payload": {"id": "bob", "label": "B"},
            "editor": None,
            "note": None,
            "ts": "2026-01-01T00:00:00.000001",
            "ip": "1.1.1.1",
        },
        {
            "type": "remove_edge",
            "payload": {"between": ["a", "b"]},
            "editor": None,
            "note": None,
            "ts": "2026-01-01T00:00:00.000002",
            "ip": "1.1.1.1",
        },
    ]
    p = tmp_path / "coauthorship_corrections.jsonl"
    write_jsonl(p, rows)

    res = import_jsonl.import_events(session, "coauthorship", p)
    assert res == {"read": 2, "inserted": 2, "skipped": 0}
    assert ev.overlay(session, "coauthorship") == fold_events(
        rows
    )  # ts preserved → same fold

    res2 = import_jsonl.import_events(session, "coauthorship", p)  # re-run
    assert res2 == {"read": 2, "inserted": 0, "skipped": 2}
    assert len(ev.read_events(session, "coauthorship")) == 2  # no duplicates


def test_import_dedupes_duplicate_ts(session, tmp_path):
    rows = [
        {
            "type": "node_label",
            "payload": {"id": "bob", "label": "B"},
            "ts": "2026-01-01T00:00:00.1",
        },
        {
            "type": "node_label",
            "payload": {"id": "bob", "label": "C"},
            "ts": "2026-01-01T00:00:00.1",
        },
    ]
    p = tmp_path / "c.jsonl"
    write_jsonl(p, rows)
    res = import_jsonl.import_events(session, "coauthorship", p)
    assert res["inserted"] == 1
    assert len(ev.read_events(session, "coauthorship")) == 1


def test_import_missing_file_is_noop(session, tmp_path):
    res = import_jsonl.import_events(session, "affiliation", tmp_path / "nope.jsonl")
    assert res == {"read": 0, "inserted": 0, "skipped": 0}


def test_chat_logs_import_is_one_shot(session, tmp_path):
    rows = [
        {
            "user_id": "u",
            "user_message": "hi",
            "bot_response": "yo",
            "timestamp": "2026-01-01T00:00:00",
            "token_count": 5,
        }
    ]
    p = tmp_path / "chat_logs.jsonl"
    write_jsonl(p, rows)
    assert import_jsonl.import_chat_logs(session, p)["inserted"] == 1
    assert (
        import_jsonl.import_chat_logs(session, p)["inserted"] == 0
    )  # guard prevents double import
