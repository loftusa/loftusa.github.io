# /// script
# requires-python = ">=3.10"
# dependencies = ["pytest", "httpx"]
# ///
"""Regression: a hand-checked "no profile" seed must survive a build_seeds.py re-run.

daniel brown / cat khor's auto-picked profiles were homonyms (a geographer, a pharmacology
researcher). Their seeds are pinned to "no profile" (s2_id=None + verified, oa_id=None +
oa_verified) — a re-run must preserve that and never call the auto-pickers for them.

    cd experiments/coauthorship && uv run tests/test_seed_preserve.py
"""
import importlib.util
import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent.parent

spec = importlib.util.spec_from_file_location("build_seeds", HERE / "build_seeds.py")
build_seeds = importlib.util.module_from_spec(spec)
sys.modules["build_seeds"] = build_seeds
spec.loader.exec_module(build_seeds)

NO_PROFILE_SEED = {
    "name": "daniel brown", "s2_id": None, "url": None, "affiliation": "",
    "homepage": None, "orcid": None, "n_papers": None, "h_index": None,
    "confidence": "high", "verified": True, "sample_titles": [], "merge_ids": [],
    "alternatives": [], "oa_id": None, "oa_url": None, "oa_affiliation": "",
    "oa_n_works": None, "oa_orcid": None, "oa_titles": [], "oa_merge_ids": [],
    "oa_alternatives": [],
    "crosscheck": {"orcid_s2": None, "orcid_oa": None, "orcid_match": False,
                   "orcid_conflict": False, "title_overlap_n": 0,
                   "title_overlap_frac": 0.0, "agree": False},
    "oa_verified": True,
}


def test_no_profile_seed_survives_rerun(tmp_path, monkeypatch):
    out = tmp_path / "seeds.json"
    out.write_text(json.dumps([NO_PROFILE_SEED]))
    monkeypatch.setattr(build_seeds, "OUT", out)
    monkeypatch.setattr(build_seeds, "NAMES", ["daniel brown"])

    def fail(*a, **k):  # any auto-pick/network call means the pin was ignored
        raise AssertionError("auto-picker called for a hand-pinned 'no profile' seed")

    for fn in ("build_entry", "build_oa", "build_oa_pinned", "search_candidates", "oa_search"):
        monkeypatch.setattr(build_seeds, fn, fail)

    build_seeds.main()

    (entry,) = json.loads(out.read_text())
    assert entry["s2_id"] is None and entry["verified"] is True
    assert entry["oa_id"] is None and entry["oa_verified"] is True
    # no homonym residue reintroduced
    assert entry["sample_titles"] == [] and entry["oa_titles"] == []
    assert entry["oa_orcid"] is None and entry["oa_affiliation"] == ""
    assert entry["crosscheck"]["agree"] is False


def test_current_seeds_are_clean():
    """The shipped seeds.json keeps the two scrubbed entries free of homonym residue."""
    seeds = json.loads((HERE / "seeds.json").read_text())
    for name in ("daniel brown", "cat khor"):
        (e,) = [s for s in seeds if s["name"] == name]
        assert e["s2_id"] is None and e["verified"] is True, name
        assert e["oa_id"] is None and e.get("oa_verified") is True, name
        assert e["sample_titles"] == [] and e["oa_titles"] == [], name
        assert e["n_papers"] is None and e["oa_orcid"] is None, name


def test_self_joined_stub_survives_rerun(tmp_path, monkeypatch):
    """merge_affiliations' stubs are preserved exactly like hand-checked no-profile seeds."""
    sys.path.insert(0, str(HERE))
    from affiliation_events import make_seed_stub
    stub = make_seed_stub("new member")
    out = tmp_path / "seeds.json"
    out.write_text(json.dumps([stub]))
    monkeypatch.setattr(build_seeds, "OUT", out)
    monkeypatch.setattr(build_seeds, "NAMES", ["new member"])

    def fail(*a, **k):
        raise AssertionError("auto-picker called for a self-joined stub")

    for fn in ("build_entry", "build_oa", "build_oa_pinned", "search_candidates", "oa_search"):
        monkeypatch.setattr(build_seeds, fn, fail)

    build_seeds.main()
    (entry,) = json.loads(out.read_text())
    assert entry["self_joined"] is True and entry["s2_id"] is None
    assert entry["oa_verified"] is True


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
