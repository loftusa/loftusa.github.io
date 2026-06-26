# /// script
# requires-python = ">=3.10"
# dependencies = ["pytest"]
# ///
"""Fold/apply/stub-sync tests for the affiliation self-service events
(affiliation_events.py + merge_affiliations.py).

    cd experiments/coauthorship && uv run tests/test_affiliation_events.py
"""
import copy
import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
from affiliation_events import (AFF_EMPTY, apply_aff_overlay, fold_aff_events,  # noqa: E402
                                make_seed_stub, norm_person, org_key)
from build_affiliations import CANON  # noqa: E402


def ev(kind, payload, ts):
    return {"type": kind, "payload": payload, "ts": ts, "editor": None, "ip": "test"}


T1, T2, T3 = "2026-06-01T00:00:00", "2026-06-02T00:00:00", "2026-06-03T00:00:00"


# ---------- fold ----------

def test_fold_lww_independent_of_input_order():
    a = ev("aff_entry_set", {"person": "can rager", "org": "Goodfire", "type": "company",
                             "role": "old"}, T1)
    b = ev("aff_entry_set", {"person": "can rager", "org": "GOODFIRE ", "type": "company",
                             "role": "new"}, T2)
    for events in ([a, b], [b, a]):
        out = fold_aff_events(events)
        assert out["entry_set"]["can rager"]["goodfire"]["role"] == "new"


def test_fold_set_then_newer_remove_cancels():
    out = fold_aff_events([
        ev("aff_entry_set", {"person": "p q", "org": "X Corp", "type": "company"}, T1),
        ev("aff_entry_remove", {"person": "p q", "org": "x corp"}, T2),
    ])
    assert out["entry_set"] == {} and out["entry_remove"]["p q"] == ["x corp"]


def test_fold_remove_then_newer_set_survives():
    out = fold_aff_events([
        ev("aff_entry_remove", {"person": "p q", "org": "X Corp"}, T1),
        ev("aff_entry_set", {"person": "p q", "org": "X Corp", "type": "company"}, T2),
    ])
    assert "x corp" in out["entry_set"]["p q"] and out["entry_remove"] == {}


def test_fold_city_lww_and_clear():
    out = fold_aff_events([
        ev("aff_city", {"person": "p q", "city": "Boston"}, T1),
        ev("aff_city", {"person": "p q", "city": ""}, T2),
    ])
    assert out["city"]["p q"] == ""


def test_fold_join_expands_and_composes_with_later_edits():
    out = fold_aff_events([
        ev("aff_join", {"name": "New Member", "city": "Lisbon",
                        "entries": [{"org": "Tiny Lab", "type": "lab", "role": "founder"}]}, T1),
        ev("aff_entry_set", {"person": "new member", "org": "Tiny Lab", "type": "lab",
                             "role": "director"}, T2),
        ev("aff_entry_remove", {"person": "new member", "org": "tiny lab"}, T3),
    ])
    assert out["join"]["new member"]["city"] == "Lisbon"
    assert out["entry_set"] == {}                     # removed last
    assert out["entry_remove"]["new member"] == ["tiny lab"]
    assert out["city"]["new member"] == "Lisbon"


def test_fold_drops_invalid_type_and_unknown_kind():
    out = fold_aff_events([
        ev("aff_entry_set", {"person": "p q", "org": "X", "type": "cult"}, T1),
        ev("aff_become_admin", {"person": "p q"}, T1),
    ])
    assert out == fold_aff_events([])


def test_fold_idempotent_and_confirm():
    events = [ev("aff_confirm", {"person": "Can Rager"}, T1)]
    once, twice = fold_aff_events(events), fold_aff_events(events + [])
    assert once == twice and once["confirmed"]["can rager"] == T1


# ---------- apply ----------

SRC = {
    "can rager": {"entries": [{"org": "Northeastern University (David Bau's lab / NDIF)",
                               "type": "lab", "role": "PhD student", "years": "2023–",
                               "current": True, "source": "https://x", "evidence": "e",
                               "verified": True}],
                  "city": "Boston", "identity_confident": True, "notes": "", "reviewed": True},
}


def overlay_with(**kw):
    ov = copy.deepcopy(AFF_EMPTY)
    ov.update(kw)
    return ov


def test_apply_edits_existing_entry_keeps_base_spelling():
    okey = org_key("northeastern university (david bau's lab / ndif)")
    ov = overlay_with(entry_set={"can rager": {okey: {
        "org": "northeastern UNIVERSITY (David Bau's Lab / NDIF)", "type": "lab",
        "role": "Postdoc", "years": "2023–", "current": True, "source": "", "ts": T2}}})
    merged, warnings = apply_aff_overlay(SRC, ov, canon=CANON)
    e = merged["can rager"]["entries"][0]
    assert e["org"] == "Northeastern University (David Bau's lab / NDIF)"  # base spelling kept
    assert e["role"] == "Postdoc" and e["verified"] is False
    assert "self-reported via site (2026-06-02)" in e["source"]
    assert SRC["can rager"]["entries"][0]["role"] == "PhD student"          # purity
    assert not warnings


def test_apply_adopts_canon_label_and_forces_type():
    ov = overlay_with(entry_set={"can rager": {"mats": {
        "org": "mats", "type": "community", "role": "", "years": "", "current": False,
        "source": "", "ts": T1}}})
    merged, _ = apply_aff_overlay(SRC, ov, canon=CANON)
    added = merged["can rager"]["entries"][-1]
    assert added["org"] == "MATS" and added["type"] == "program"   # canon label + its type win


def test_apply_remove_drops_hand_entry_but_not_base_file():
    okey = org_key("Northeastern University (David Bau's lab / NDIF)")
    merged, _ = apply_aff_overlay(SRC, overlay_with(entry_remove={"can rager": [okey]}),
                                  canon=CANON)
    assert merged["can rager"]["entries"] == [] and len(SRC["can rager"]["entries"]) == 1


def test_apply_join_creates_record_and_orphan_edits_warn():
    ov = overlay_with(
        join={"new member": {"name": "New Member", "city": "Lisbon", "scholar_url": None,
                             "homepage": "https://new.me", "ts": T1}},
        entry_set={"ghost person": {"x": {"org": "X", "type": "company", "role": "",
                                          "years": "", "current": False, "source": "", "ts": T1}}})
    merged, warnings = apply_aff_overlay(SRC, ov, canon=CANON)
    rec = merged["new member"]
    assert rec["city"] == "Lisbon" and "self-joined via site (2026-06-01)" in rec["notes"]
    assert "https://new.me" in rec["notes"]
    assert any("ghost person" in w for w in warnings)


def test_apply_join_collision_skipped():
    ov = overlay_with(join={"can rager": {"name": "Can Rager", "city": "", "scholar_url": None,
                                          "homepage": None, "ts": T1}})
    merged, warnings = apply_aff_overlay(SRC, ov, canon=CANON)
    assert merged["can rager"]["reviewed"] is True            # untouched original record
    assert any("collides" in w for w in warnings)


def test_apply_entry_cap():
    base = {"p q": {"entries": [], "city": "", "identity_confident": True, "notes": "",
                    "reviewed": False}}
    sets = {f"org {i:02d}": {"org": f"Org {i:02d}", "type": "company", "role": "", "years": "",
                             "current": False, "source": "", "ts": T1} for i in range(15)}
    merged, warnings = apply_aff_overlay(base, overlay_with(entry_set={"p q": sets}), canon={})
    assert len(merged["p q"]["entries"]) == 12
    assert sum("cap" in w for w in warnings) == 3


def test_apply_deterministic():
    ov = overlay_with(entry_set={"can rager": {
        "b org": {"org": "B Org", "type": "company", "role": "", "years": "", "current": False,
                  "source": "", "ts": T1},
        "a org": {"org": "A Org", "type": "company", "role": "", "years": "", "current": False,
                  "source": "", "ts": T1}}})
    m1, _ = apply_aff_overlay(SRC, ov, canon=CANON)
    m2, _ = apply_aff_overlay(SRC, ov, canon=CANON)
    assert json.dumps(m1, sort_keys=True) == json.dumps(m2, sort_keys=True)
    assert [e["org"] for e in m1["can rager"]["entries"][1:]] == ["A Org", "B Org"]


# ---------- stub sync (merge_affiliations.sync_stubs) ----------

def test_stub_sync_round_trip():
    from merge_affiliations import sync_stubs
    seeds = [{"name": "can rager", "s2_id": "123", "verified": True}]
    roster = {"core": ["can rager"], "self_joined": []}
    ov = overlay_with(join={"new member": {"name": "New Member", "city": "", "scholar_url": None,
                                           "homepage": None, "ts": T1}})
    added, removed = sync_stubs(ov, seeds, roster)
    assert added == ["new member"] and removed == []
    assert seeds[-1]["self_joined"] is True and seeds[-1]["s2_id"] is None
    assert roster["self_joined"] == ["new member"]

    # revoke the join (admin DELETE) -> next sync removes ONLY the tagged stub
    added, removed = sync_stubs(overlay_with(), seeds, roster)
    assert added == [] and removed == ["new member"]
    assert [s["name"] for s in seeds] == ["can rager"]
    assert roster["self_joined"] == []


def test_stub_sync_never_adopts_unmanaged_collision():
    from merge_affiliations import sync_stubs
    seeds = [{"name": "can rager", "s2_id": "123", "verified": True}]
    roster = {"core": ["can rager"], "self_joined": []}
    ov = overlay_with(join={"can rager": {"name": "Can Rager", "city": "", "scholar_url": None,
                                          "homepage": None, "ts": T1}})
    added, removed = sync_stubs(ov, seeds, roster)
    assert added == [] and removed == [] and len(seeds) == 1


def test_seed_stub_shape_matches_no_profile_convention():
    stub = make_seed_stub("new member", homepage="https://new.me")
    assert stub["s2_id"] is None and stub["verified"] is True
    assert stub["oa_id"] is None and stub["oa_verified"] is True
    assert stub["self_joined"] is True and stub["homepage"] == "https://new.me"


def test_apply_remove_matches_canonical_label():
    """The UI shows CANON labels; remove must drop the raw-variant base entry (was silent no-op)."""
    gone = [org_key("Bau Lab (Northeastern / NDIF)")]      # canonical label of can's raw entry
    merged, _ = apply_aff_overlay(SRC, overlay_with(entry_remove={"can rager": gone}), canon=CANON)
    assert merged["can rager"]["entries"] == []


def test_apply_edit_matches_canonical_label_no_duplicate():
    """Editing via the canonical label updates the raw-variant entry instead of appending a twin."""
    okey = org_key("Bau Lab (Northeastern / NDIF)")
    ov = overlay_with(entry_set={"can rager": {okey: {
        "org": "Bau Lab (Northeastern / NDIF)", "type": "lab", "role": "Alum", "years": "",
        "current": False, "source": "", "ts": T2}}})
    merged, _ = apply_aff_overlay(SRC, ov, canon=CANON)
    assert len(merged["can rager"]["entries"]) == 1
    e = merged["can rager"]["entries"][0]
    assert e["role"] == "Alum"
    assert e["org"] == "Northeastern University (David Bau's lab / NDIF)"   # raw spelling kept


def test_apply_two_new_spellings_converge_no_slug_collision():
    """Two people typing variants of the same NEW org must adopt one spelling (slug-space dedupe),
    or the build's slug-collision assert would crash the nightly."""
    base = {"a a": {"entries": [], "city": "", "identity_confident": True, "notes": "", "reviewed": False},
            "b b": {"entries": [], "city": "", "identity_confident": True, "notes": "", "reviewed": False}}
    mk = lambda org: {org_key(org): {"org": org, "type": "company", "role": "", "years": "",
                                     "current": False, "source": "", "ts": T1}}
    ov = overlay_with(entry_set={"a a": mk("Acme AI"), "b b": mk("Acme  AI.")})
    merged, warnings = apply_aff_overlay(base, ov, canon={})
    orgs = {merged["a a"]["entries"][0]["org"], merged["b b"]["entries"][0]["org"]}
    assert orgs == {"Acme AI"}, orgs                    # first (sorted) spelling adopted by both
    assert sum("needs a CANON line" in w for w in warnings) == 1


def test_apply_edit_locks_type_of_shared_org():
    """A type-changing edit of an existing entry keeps the established type (build asserts one
    type per org slug across all holders)."""
    okey = org_key("Northeastern University (David Bau's lab / NDIF)")
    ov = overlay_with(entry_set={"can rager": {okey: {
        "org": "Northeastern University (David Bau's lab / NDIF)", "type": "community",
        "role": "x", "years": "", "current": False, "source": "", "ts": T2}}})
    merged, _ = apply_aff_overlay(SRC, ov, canon=CANON)
    assert merged["can rager"]["entries"][0]["type"] == "lab"


def test_build_runs_with_nonempty_overlay(tmp_path, monkeypatch):
    """The untested seam: a realistic overlay (join + canon-variant edit + remove + two-spelling
    new org) must survive a full build_affiliations.main() run."""
    import build_affiliations as ba
    events = [
        ev("aff_join", {"name": "New Member", "city": "Lisbon",
                        "entries": [{"org": "Acme AI", "type": "company"}]}, T1),
        ev("aff_entry_set", {"person": "ted kyi", "org": "acme  ai.", "type": "company"}, T2),
        ev("aff_entry_set", {"person": "can rager", "org": "Bau Lab (Northeastern / NDIF)",
                             "type": "lab", "role": "Alum"}, T2),
        ev("aff_entry_remove", {"person": "ted kyi", "org": "Deep Sentinel"}, T2),
    ]
    ovf = tmp_path / "affiliation_overrides.json"
    ovf.write_text(json.dumps(fold_aff_events(events)))
    # in the nightly, merge_affiliations' seeds stub makes build_graph mint the joiner's node
    # BEFORE build_affiliations runs — synthesize that precondition
    graph = json.loads(ba.GRAPH.read_text())
    graph["nodes"].append({"id": "new member", "label": "New Member", "initials": "NM",
                           "community": -1, "is_list": True, "x": 0.0, "y": 0.0})
    gf = tmp_path / "coauthorship.json"
    gf.write_text(json.dumps(graph))
    # ...and merge_affiliations syncs the join into the roster before the build runs
    roster = json.loads((HERE / "roster.json").read_text())
    roster["self_joined"] = ["new member"]
    rf = tmp_path / "roster.json"
    rf.write_text(json.dumps(roster))
    monkeypatch.setattr(ba, "ROSTER_PATH", rf)
    monkeypatch.setattr(ba, "GRAPH", gf)
    monkeypatch.setattr(ba, "OVERRIDES", ovf)
    monkeypatch.setattr(ba, "OUT", tmp_path / "affiliations.json")
    monkeypatch.setattr(ba, "SHARED_OUT", tmp_path / "shared.json")
    monkeypatch.setattr(ba, "PERSON_PAGES_DIR", tmp_path / "networks")
    ba.main()                                           # must not raise (slug/type asserts inside)
    out = json.loads((tmp_path / "affiliations.json").read_text())
    people = {p["id"] for p in out["people"]}
    n_src = len(json.loads((Path(ba.__file__).parent / "affiliations.json").read_text()))
    assert "new member" in people and len(people) == n_src + 1
    labels = [o["label"] for o in out["orgs"]]
    assert labels.count("Acme AI") == 1 and "Acme  AI." not in labels
    assert "Deep Sentinel" not in labels                # ted was its only member
    acme = next(o for o in out["orgs"] if o["label"] == "Acme AI")
    assert acme["n_members"] == 2


# ---------- end-to-end round trip: join event -> build sees the person -> revert -> gone ----------

def test_vandal_round_trip_through_build_merge():
    import build_affiliations as ba
    events = [ev("aff_join", {"name": "Totally Fake", "city": "Nowhere",
                              "entries": [{"org": "Fake Org", "type": "company"}]}, T1)]
    overlay = fold_aff_events(events)
    src = json.loads(ba.SRC.read_text())
    merged, _ = apply_aff_overlay(src, overlay, canon=CANON)
    assert "totally fake" in merged and len(merged) == len(src) + 1
    # admin deletes the event -> empty overlay -> person gone
    merged2, _ = apply_aff_overlay(src, fold_aff_events([]), canon=CANON)
    assert "totally fake" not in merged2 and len(merged2) == len(src)


def test_registry_reconcile_adds_empty_and_rejects_strays(tmp_path):
    from registry import reconcile_membership
    roster = tmp_path / "roster.json"
    roster.write_text(json.dumps({"core": ["can rager", "brand-new person"], "self_joined": []}))
    records = {"can rager": {"entries": [], "city": "", "identity_confident": True,
                             "notes": "", "reviewed": True}}
    records, added = reconcile_membership(records, roster)
    assert added == ["brand-new person"]
    assert records["brand-new person"]["entries"] == []
    assert "no recorded chapters yet" in records["brand-new person"]["notes"]

    stray = {"can rager": records["can rager"], "smuggled in": {"entries": []}}
    with pytest.raises(AssertionError, match="smuggled"):
        reconcile_membership(stray, roster)


def test_registry_roster_covers_current_data():
    """The committed roster and the committed affiliations source agree exactly."""
    from registry import load_roster
    names = load_roster()
    src = json.loads((HERE / "affiliations.json").read_text())
    assert {norm_person(n) for n in names} == {norm_person(k) for k in src}


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
