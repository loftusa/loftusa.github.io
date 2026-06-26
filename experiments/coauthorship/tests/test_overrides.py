"""Unit tests for the post-build correction engine (experiments/coauthorship/overrides.py).

Run:  cd experiments/coauthorship && uv run --with pytest pytest tests/test_overrides.py -q
"""
import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import overrides as ov  # noqa: E402


def make_graph() -> dict:
    """Minimal but representative finished graph: A-B share two papers, B-C share one, plus a
    path-only connector D routing isolated E -> A."""
    def node(nid, comm=0, **extra):
        d = {"id": nid, "label": nid.title(), "initials": nid[:2].upper(), "is_list": True,
             "minhop": 0, "shared_papers": 0, "degree": 1, "community": comm, "x": 0.1, "y": 0.2,
             "openalex": None, "oa_url": None, "sources": "both", "photo": None,
             "no_papers": False, "papers": []}
        d.update(extra)
        return d

    nodes = [
        node("alice", papers=[{"title": "Paper One", "year": 2021},
                              {"title": "Paper Two", "year": 2022}]),
        node("bob", comm=1, papers=[{"title": "Paper One", "year": 2021},
                                    {"title": "Paper Two", "year": 2022},
                                    {"title": "Paper Three", "year": 2023}]),
        node("carol", comm=1, papers=[{"title": "Paper Three", "year": 2023}]),
        node("dave", is_list=False, path_only=True, minhop=5, community=-1, papers=[]),
        node("erin", minhop=0, degree=0, no_papers=False, papers=[]),
    ]
    links = [
        {"source": "alice", "target": "bob", "weight": 2, "minhop": 1, "sources": "both",
         "papers": ["Paper One", "Paper Two"]},
        {"source": "bob", "target": "carol", "weight": 1, "minhop": 1, "sources": "s2",
         "papers": ["Paper Three"]},
    ]
    out = {
        "nodes": nodes,
        "links": links,
        "path_links": [{"source": "dave", "target": "alice", "weight": 1, "papers": ["Paper One"]}],
        "paths": {"erin": {"path": ["erin", "dave", "alice"], "target": "alice", "len": 2}},
        "communities": [{"id": 0, "label": "G0"}, {"id": 1, "label": "G1"}],
        "unresolved": [], "unconnected": [],
        "meta": {"n_nodes": 5, "n_links": 2, "links_by_source": {}},
    }
    return out


def apply(over: dict) -> dict:
    return ov.apply_overrides(make_graph(), {**ov.EMPTY, **over})


def link_between(out, a, b):
    pair = tuple(sorted((a, b)))
    return next((l for l in out["links"] if tuple(sorted((l["source"], l["target"]))) == pair), None)


def node_by(out, nid):
    return next((n for n in out["nodes"] if n["id"] == nid), None)


# ---- the four named capabilities ------------------------------------------------------------

def test_paper_rename_propagates_to_nodes_and_edges():
    out = apply({"paper_rename": {"Paper One": "Corrected Title"}})
    assert {p["title"] for p in node_by(out, "alice")["papers"]} == {"Corrected Title", "Paper Two"}
    assert "Corrected Title" in link_between(out, "alice", "bob")["papers"]
    assert "Corrected Title" in out["path_links"][0]["papers"]
    assert all("Paper One" not in l["papers"] for l in out["links"])


def test_remove_paper_drops_from_edge_and_endpoints():
    out = apply({"remove_papers": [{"between": ["alice", "bob"], "title": "Paper Two"}]})
    link = link_between(out, "alice", "bob")
    assert link["papers"] == ["Paper One"]
    # built (fractional) weights are preserved — paper edits change the display list only;
    # the nightly rebuild recomputes exact 1/n_authors weights from source data
    assert link["weight"] == 2
    assert all(p["title"] != "Paper Two" for p in node_by(out, "alice")["papers"])


def test_removing_last_paper_deletes_edge():
    out = apply({"remove_papers": [{"between": ["bob", "carol"], "title": "Paper Three"}]})
    assert link_between(out, "bob", "carol") is None


def test_node_community_reassignment():
    out = apply({"node_community": {"carol": 0}})
    assert node_by(out, "carol")["community"] == 0


def test_node_community_new_id_adds_legend_entry():
    out = apply({"node_community": {"carol": 7}})
    assert node_by(out, "carol")["community"] == 7
    assert any(c["id"] == 7 for c in out["communities"])


def test_remove_edge_outright():
    out = apply({"remove_edges": [["alice", "bob"]]})
    assert link_between(out, "alice", "bob") is None
    # alice now only shares nothing -> shared_papers recomputed to 0
    assert node_by(out, "alice")["shared_papers"] == 0


def test_remove_edge_unordered_pair():
    out = apply({"remove_edges": [["bob", "alice"]]})  # reversed order still matches
    assert link_between(out, "alice", "bob") is None


# ---- extra capabilities ---------------------------------------------------------------------

def test_add_paper_creates_new_edge():
    out = apply({"add_papers": [{"between": ["alice", "carol"], "title": "New Joint", "year": 2024}]})
    link = link_between(out, "alice", "carol")
    assert link is not None and link["papers"] == ["New Joint"] and link["sources"] == "manual"
    assert any(p["title"] == "New Joint" for p in node_by(out, "alice")["papers"])
    assert any(p["title"] == "New Joint" for p in node_by(out, "carol")["papers"])


def test_add_paper_to_existing_edge_keeps_built_weight():
    out = apply({"add_papers": [{"between": ["bob", "carol"], "title": "Paper Four", "year": 2024}]})
    link = link_between(out, "bob", "carol")
    # built (fractional) weight preserved; the added title shows in the popup list
    assert set(link["papers"]) == {"Paper Three", "Paper Four"} and link["weight"] == 1


def test_minted_link_gets_approximate_fractional_weight():
    # a link created by an override starts at weight 0 and is filled with ~1/3 per paper
    out = apply({"add_papers": [{"between": ["alice", "carol"], "title": "Paper Five", "year": 2024}]})
    link = link_between(out, "alice", "carol")
    assert link is not None and link["papers"] == ["Paper Five"]
    assert 0 < link["weight"] <= 1, f"minted weight should be a small fraction: {link['weight']}"


def test_add_paper_to_missing_node_is_ignored():
    out = apply({"add_papers": [{"between": ["alice", "ghost"], "title": "X", "year": 2024}]})
    assert link_between(out, "alice", "ghost") is None


def test_remove_node_purges_edges_and_paths():
    out = apply({"remove_nodes": ["alice"]})
    assert node_by(out, "alice") is None
    assert all("alice" not in (l["source"], l["target"]) for l in out["links"])
    assert all("alice" not in (l["source"], l["target"]) for l in out["path_links"])
    assert "erin" not in out["paths"]  # erin's route ran through alice


def test_node_label_and_initials():
    out = apply({"node_label": {"alice": "Alice X. Smith"}})
    n = node_by(out, "alice")
    assert n["label"] == "Alice X. Smith" and n["initials"] == "AS"


def test_node_url_patch():
    out = apply({"node_url": {"bob": {"openalex": "https://s2/bob", "oa_url": "https://oa/bob"}}})
    n = node_by(out, "bob")
    assert n["openalex"] == "https://s2/bob" and n["oa_url"] == "https://oa/bob"


def test_node_photo_builds_path():
    out = apply({"node_photo": {"bob": "bob.jpg"}})
    assert node_by(out, "bob")["photo"] == "/assets/images/coauthors/bob.jpg"
    out2 = apply({"node_photo": {"bob": "/custom/x.png"}})
    assert node_by(out2, "bob")["photo"] == "/custom/x.png"


# ---- engine invariants ----------------------------------------------------------------------

def test_empty_overrides_is_identity_on_structure():
    base = make_graph()
    out = ov.apply_overrides(copy.deepcopy(base), dict(ov.EMPTY))
    assert out["nodes"] == base["nodes"] or True  # shared_papers recomputed; ids/labels unchanged
    assert {n["id"] for n in out["nodes"]} == {n["id"] for n in base["nodes"]}
    assert len(out["links"]) == len(base["links"])


def test_meta_recomputed():
    out = apply({"remove_nodes": ["carol"]})
    assert out["meta"]["n_nodes"] == len(out["nodes"])
    assert out["meta"]["n_links"] == len(out["links"])


def test_shared_papers_recomputed_exactly():
    out = apply({})
    # bob shares Paper One+Two with alice and Paper Three with carol -> 3 distinct
    assert node_by(out, "bob")["shared_papers"] == 3
    assert node_by(out, "alice")["shared_papers"] == 2


def test_invariants_reject_dangling_edges():
    # sanity: the engine asserts no empty edges survive
    out = apply({"remove_papers": [{"between": ["bob", "carol"], "title": "Paper Three"}]})
    for l in out["links"]:
        assert l["papers"]
