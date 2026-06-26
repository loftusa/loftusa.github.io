"""Unit tests for the event-log folding (experiments/coauthorship/merge_corrections.py).

Run:  cd experiments/coauthorship && uv run --with pytest pytest tests/test_merge_corrections.py -q
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import merge_corrections as mc  # noqa: E402


def ev(t, payload, ts):
    return {"type": t, "payload": payload, "ts": ts}


def test_last_write_wins_on_node_label():
    out = mc.fold_events([
        ev("node_label", {"id": "bob", "label": "Bob One"}, "2026-01-01T00:00:00"),
        ev("node_label", {"id": "bob", "label": "Bob Two"}, "2026-01-02T00:00:00"),
    ])
    assert out["node_label"]["bob"] == "Bob Two"


def test_last_write_wins_independent_of_input_order():
    later = ev("node_community", {"id": "bob", "community": 2}, "2026-02-01T00:00:00")
    earlier = ev("node_community", {"id": "bob", "community": 1}, "2026-01-01T00:00:00")
    assert mc.fold_events([later, earlier])["node_community"]["bob"] == 2
    assert mc.fold_events([earlier, later])["node_community"]["bob"] == 2


def test_idempotent():
    events = [
        ev("paper_rename", {"old": "X", "new": "Y"}, "2026-01-01T00:00:00"),
        ev("remove_edge", {"between": ["alice", "bob"]}, "2026-01-02T00:00:00"),
        ev("add_paper", {"between": ["carol", "dave"], "title": "Z", "year": 2025}, "2026-01-03T00:00:00"),
    ]
    a = mc.fold_events(events)
    b = mc.fold_events(events)
    assert a == b


def test_remove_edge_deduped_by_pair_unordered():
    out = mc.fold_events([
        ev("remove_edge", {"between": ["alice", "bob"]}, "2026-01-01T00:00:00"),
        ev("remove_edge", {"between": ["bob", "alice"]}, "2026-01-02T00:00:00"),
    ])
    assert out["remove_edges"] == [["alice", "bob"]]


def test_add_then_newer_remove_cancels_the_add():
    out = mc.fold_events([
        ev("add_paper", {"between": ["a", "b"], "title": "Ghost", "year": 2025}, "2026-01-01T00:00:00"),
        ev("remove_paper", {"between": ["a", "b"], "title": "Ghost"}, "2026-01-02T00:00:00"),
    ])
    assert out["add_papers"] == []
    # the remove is still emitted (harmless no-op if the title isn't in the base graph, correct if it is)
    assert out["remove_papers"] == [{"between": ["a", "b"], "title": "Ghost"}]


def test_remove_then_newer_add_keeps_the_add():
    out = mc.fold_events([
        ev("remove_paper", {"between": ["a", "b"], "title": "Real"}, "2026-01-01T00:00:00"),
        ev("add_paper", {"between": ["a", "b"], "title": "Real", "year": 2025}, "2026-01-02T00:00:00"),
    ])
    assert [i["title"] for i in out["add_papers"]] == ["Real"]
    assert out["remove_papers"] == []


def test_remove_paper_on_base_graph_is_emitted():
    out = mc.fold_events([
        ev("remove_paper", {"between": ["a", "b"], "title": "WrongPaper"}, "2026-01-01T00:00:00"),
    ])
    assert out["remove_papers"] == [{"between": ["a", "b"], "title": "WrongPaper"}]


def test_remove_node_is_sticky_and_normalized():
    out = mc.fold_events([ev("remove_node", {"id": "Roy Rinberg"}, "2026-01-01T00:00:00")])
    assert out["remove_nodes"] == ["roy rinberg"]


def test_node_url_merges_fields():
    out = mc.fold_events([
        ev("node_url", {"id": "bob", "openalex": "https://s2/bob"}, "2026-01-01T00:00:00"),
        ev("node_url", {"id": "bob", "oa_url": "https://oa/bob"}, "2026-01-02T00:00:00"),
    ])
    # LWW per key: the later event replaces the whole node_url entry (documented semantics)
    assert out["node_url"]["bob"] == {"oa_url": "https://oa/bob"}


def test_unknown_event_ignored():
    out = mc.fold_events([ev("explode", {"id": "x"}, "2026-01-01T00:00:00")])
    assert out == mc.fold_events([])


def test_fold_output_is_valid_overrides_input():
    """The folded result must be consumable by apply_overrides without KeyErrors."""
    import overrides as ovr
    folded = mc.fold_events([
        ev("node_label", {"id": "z", "label": "Zed"}, "2026-01-01T00:00:00"),
    ])
    # every contract key present
    assert set(ovr.EMPTY) <= set(folded)
