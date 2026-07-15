# /// script
# requires-python = ">=3.10"
# dependencies = ["pytest"]
# ///
"""Sanity + determinism checks for build_affiliations.py and the shipped affiliations data.

    cd experiments/coauthorship && uv run tests/test_affiliations_data.py
"""
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent.parent
REPO = HERE.parents[1]
OUT = REPO / "public" / "assets" / "data" / "affiliations.json"
SHARED_OUT = REPO / "public" / "assets" / "data" / "analyses-affiliations" / "shared.json"

sys.path.insert(0, str(HERE))
from build_affiliations import PARENT, TYPE_WEIGHT, load_src, norm, normalize_years, slug  # noqa: E402


@pytest.fixture(scope="module")
def data():
    return json.loads(OUT.read_text())


def test_build_is_deterministic():
    before, before_shared = OUT.read_bytes(), SHARED_OUT.read_bytes()
    try:
        subprocess.run([sys.executable, str(HERE / "build_affiliations.py")], check=True,
                       capture_output=True)
        assert OUT.read_bytes() == before, "re-running the build changed affiliations.json"
        assert SHARED_OUT.read_bytes() == before_shared, "re-running the build changed shared.json"
    finally:
        OUT.write_bytes(before)            # never leave the shipped artifacts dirty
        SHARED_OUT.write_bytes(before_shared)


def test_people_complete(data):
    """People ship under the map's canonical node ids; the source names map onto them 1:1."""
    graph = json.loads((REPO / "public" / "assets" / "data" / "coauthorship.json").read_text())
    listed = {n["id"] for n in graph["nodes"] if n.get("is_list")}
    ids = {p["id"] for p in data["people"]}
    assert ids <= listed, ids - listed
    src = load_src()                       # the overlay-merged source the build actually uses
    assert {norm(s) for s in src} == {norm(i) for i in ids}
    assert len(data["people"]) == len(src) >= 48


def test_analyses_shared_consistent(data):
    shared = json.loads(SHARED_OUT.read_text())
    ids = {p["id"] for p in data["people"]}
    assert set(shared["nodes"]) == ids
    assert shared["communities"] == data["communities"]
    from collections import defaultdict
    deg = defaultdict(int)
    for a, b, w in shared["links"]:
        assert a in ids and b in ids and a < b
        deg[a] += 1
        deg[b] += 1
    assert len(shared["links"]) == len(data["projection"])
    for nid, n in shared["nodes"].items():
        assert n["degree"] == deg[nid], nid
        assert n["is_list"] is True and isinstance(n["label"], str)


def test_links_consistent(data):
    pids = {p["id"] for p in data["people"]}
    orgs = {o["id"]: o for o in data["orgs"]}
    member_counts = defaultdict(int)
    seen = set()
    for l in data["links"]:
        assert l["person"] in pids and l["org"] in orgs
        assert (l["person"], l["org"]) not in seen, f"duplicate link {l['person']} -> {l['org']}"
        seen.add((l["person"], l["org"]))
        member_counts[l["org"]] += 1
    for o in orgs.values():
        assert o["type"] in TYPE_WEIGHT
        assert member_counts[o["id"]] == o["n_members"], o["id"]


def test_canonicalization_merged_the_big_labs(data):
    """The whole point: lab variants must collapse to single hub nodes."""
    by_label = {o["label"]: o for o in data["orgs"]}
    assert by_label["NeuroData Lab (Johns Hopkins)"]["n_members"] >= 10
    assert by_label["Bau Lab (Northeastern / NDIF)"]["n_members"] >= 8
    assert by_label["MATS"]["n_members"] >= 7
    # no stray un-merged variants survive
    strays = [o["label"] for o in data["orgs"]
              if ("neurodata" in o["label"].lower() or "bau" in o["label"].lower())
              and o["label"] not in ("NeuroData Lab (Johns Hopkins)", "Bau Lab (Northeastern / NDIF)")]
    assert not strays, strays


def test_projection_consistent(data):
    oid_members = {o["id"]: o["n_members"] for o in data["orgs"]}
    types = {o["id"]: o["type"] for o in data["orgs"]}
    parent_of = {slug(c): slug(p) for c, p in PARENT.items()}
    assert data["meta"]["type_weights"] == TYPE_WEIGHT
    for p in data["projection"]:
        assert p["a"] < p["b"]
        assert p["shared"], "projected pair with no shared org"
        assert all(oid_members[o] >= 2 for o in p["shared"])
        assert p["weight"] == round(sum(TYPE_WEIGHT[types[o]] for o in p["shared"]), 1)
        # nesting discount: a shared lab must have displaced its university from the pair
        for o in p["shared"]:
            assert parent_of.get(o) not in p["shared"], \
                f"{p['a']}–{p['b']} shares both {o} and its parent {parent_of[o]}"


@pytest.mark.parametrize("specs,want", [
    ([("2018–2021", False)], "2018–2021"),            # closed span passes through
    ([("2024–", True)], "2024–"),                     # ongoing, current
    ([("2019–", False)], "2019"),                     # dash but NOT current = started, end unknown
    ([("2024", False)], "2024"),                      # single year
    ([("–2027 term", False)], "–2027"),               # end-only, noise stripped
    ([("2019–c. 2024/25", False)], "2019–2024"),      # fuzzy end normalized
    ([("", True)], ""),                               # undated stays undated
    ([("2015–2019", False), ("2024–", True)], "2015–"),   # merged stints: earliest start, ongoing wins
    ([("2024–", True), ("2022–", True)], "2022–"),        # fred's Harvard schools
])
def test_normalize_years(specs, want):
    assert normalize_years(specs) == want


def test_panels_fresh():
    """The six affiliation panels' committed JSON must match a fresh re-run — this is the only
    gate that re-fires every panel's internal headline asserts after a data edit."""
    panel_dir = HERE / "analyses-affiliations"
    out_dir = REPO / "public" / "assets" / "data" / "analyses-affiliations"
    scripts = sorted(panel_dir.glob("*.py"))
    assert len(scripts) == 7, [s.name for s in scripts]
    for script in scripts:
        out = out_dir / (script.stem + ".json")
        before = out.read_bytes()
        try:
            r = subprocess.run([sys.executable, str(script)], check=False, capture_output=True)
            assert r.returncode == 0, f"{script.name} failed:\n{r.stderr.decode()[-800:]}"
            assert out.read_bytes() == before, \
                f"{script.name}: committed JSON is stale — re-run the panel scripts and recheck prose"
        finally:
            out.write_bytes(before)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
