"""Wire-compat core: the SQLite event store must fold to byte-identical overlays
as the pre-migration JSONL path. We assert against the REAL fold_* functions so the
store can never silently diverge from what the nightly baker consumes.
"""
import json

from backend.app.services import events as ev
from experiments.coauthorship.affiliation_events import fold_aff_events
from experiments.coauthorship.overrides import fold_events


def append(session, family, type_, payload, editor=None, note=None, ip="1.2.3.4"):
    return ev.append_event(
        session,
        family=family,
        type=type_,
        payload=payload,
        editor=editor,
        note=note,
        ip=ip,
    )


def test_stored_event_has_canonical_shape(session):
    e = append(
        session,
        "coauthorship",
        "node_label",
        {"id": "bob", "label": "B"},
        editor="ed",
        note="n",
        ip="9.9.9.9",
    )
    assert set(e) == {"type", "payload", "editor", "note", "ts", "ip"}
    assert e["type"] == "node_label"
    assert e["payload"] == {"id": "bob", "label": "B"}
    assert e["editor"] == "ed" and e["note"] == "n" and e["ip"] == "9.9.9.9"


def test_overlay_matches_fold_events_exactly(session):
    stored = [
        append(
            session,
            "coauthorship",
            "node_label",
            {"id": "Bob Smith", "label": "Bob Q. Smith"},
        ),
        append(session, "coauthorship", "remove_edge", {"between": ["alice", "bob"]}),
    ]
    overlay = ev.overlay(session, "coauthorship")
    assert overlay == fold_events(stored)  # never diverge from the real fold
    assert overlay["node_label"]["bob smith"] == "Bob Q. Smith"
    assert overlay["remove_edges"] == [["alice", "bob"]]


def test_empty_overlay_is_empty_contract(session):
    assert ev.overlay(session, "coauthorship") == fold_events([])


def test_append_assigns_distinct_increasing_ts(session):
    a = append(session, "coauthorship", "remove_node", {"id": "x"})
    b = append(session, "coauthorship", "remove_node", {"id": "y"})
    assert a["ts"] != b["ts"]
    assert a["ts"] < b["ts"]


def test_delete_by_ts_durably_removes(session):
    vandal = append(
        session, "coauthorship", "node_label", {"id": "bob", "label": "Vandal"}
    )
    append(session, "coauthorship", "remove_edge", {"between": ["alice", "bob"]})
    res = ev.delete_by_ts(session, "coauthorship", vandal["ts"])
    assert res["ok"] and res["removed"] == 1 and res["remaining"] == 1
    ov = ev.overlay(session, "coauthorship")
    assert ov["node_label"] == {}  # vandal gone from the live overlay
    assert ov["remove_edges"] == [["alice", "bob"]]  # unrelated event untouched


def test_delete_unknown_ts_removes_zero(session):
    append(session, "coauthorship", "node_label", {"id": "bob", "label": "B"})
    res = ev.delete_by_ts(session, "coauthorship", "1999-01-01T00:00:00")
    assert res["removed"] == 0


def test_raw_jsonl_is_parseable_and_carries_audit_fields(session):
    append(
        session,
        "coauthorship",
        "node_label",
        {"id": "bob", "label": "B"},
        editor="alice",
        ip="9.9.9.9",
    )
    text = ev.raw_jsonl(session, "coauthorship")
    lines = [ln for ln in text.splitlines() if ln.strip()]
    assert len(lines) == 1
    row = json.loads(lines[0])
    assert row["type"] == "node_label" and "ts" in row
    assert row["editor"] == "alice" and row["ip"] == "9.9.9.9"


def test_byte_cap_rejects_oversized_event(session):
    import pytest
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as exc:
        append(
            session,
            "coauthorship",
            "node_label",
            {"id": "bob", "label": "B", "junk": "A" * 8000},
        )
    assert exc.value.status_code == 413


# ----- affiliation family folds with the affiliation fold function ----------------------------


def test_affiliation_overlay_matches_fold(session):
    stored = [
        append(session, "affiliation", "aff_confirm", {"person": "can rager"}),
        append(
            session,
            "affiliation",
            "aff_city",
            {"person": "can rager", "city": "Boston"},
        ),
    ]
    ov = ev.overlay(session, "affiliation")
    assert ov == fold_aff_events(stored)
    assert "can rager" in ov["confirmed"] and ov["city"]["can rager"] == "Boston"


def test_families_are_isolated(session):
    append(session, "coauthorship", "remove_node", {"id": "x"})
    append(session, "affiliation", "aff_confirm", {"person": "p"})
    assert ev.overlay(session, "coauthorship")["remove_nodes"] == ["x"]
    assert "p" in ev.overlay(session, "affiliation")["confirmed"]
