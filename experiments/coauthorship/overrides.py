"""Apply hand/crowd corrections to the finished co-authorship graph.

The graph in ``assets/data/coauthorship.json`` is rebuilt nightly from Semantic Scholar +
OpenAlex, so it carries real errors about real people. ``overrides.json`` is the durable,
git-tracked record of human corrections; :func:`apply_overrides` replays it onto the finished
``out`` dict as the very last build step, so corrections survive every rebuild without touching
the (fragile) graph/hop/clustering algorithms upstream.

This module is intentionally dependency-free and operates only on plain dicts/lists, so it can be
unit-tested without running the whole build.

Override schema (``overrides.json``)::

    {
      "version": 1,
      "node_label":     {"<id>": "Display Name"},
      "node_community": {"<id>": 2},
      "node_url":       {"<id>": {"openalex": "...", "oa_url": "..."}},
      "node_photo":     {"<id>": "file.jpg"},          # dropped into assets/images/coauthors/
      "remove_nodes":   ["<id>", ...],                  # opt-out: drop node + its edges/paths
      "paper_rename":   {"old title": "Correct Title"}, # renamed everywhere it appears
      "remove_papers":  [{"between": ["a", "b"], "title": "..."}],
      "add_papers":     [{"between": ["a", "b"], "title": "...", "year": 2025}],
      "remove_edges":   [["a", "b"], ...]               # delete an edge outright
    }

``<id>`` is the normalized lowercase node id used as the graph key (e.g. ``"can rager"``).
Edges are an unordered name pair. ``apply_overrides`` is order-independent of how the file lists
keys: it always runs the ops in a fixed, safe sequence (remove nodes -> remove edges/papers ->
add papers -> rename -> node fields -> recompute derived fields).
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

PHOTO_DIR_URL = "/assets/images/coauthors"
EMPTY: dict = {
    "version": 1,
    "node_label": {},
    "node_community": {},
    "node_url": {},
    "node_photo": {},
    "remove_nodes": [],
    "paper_rename": {},
    "remove_papers": [],
    "add_papers": [],
    "remove_edges": [],
}


def _norm_title(s: str) -> str:
    """Case/whitespace-insensitive title key. Punctuation is kept (colons distinguish papers)."""
    return " ".join((s or "").split()).casefold()


def _norm_id(s: str) -> str:
    """Match an override id against node ids: lowercase, collapse whitespace."""
    return " ".join((s or "").split()).casefold()


def _initials(label: str) -> str:
    parts = [p for p in re.split(r"\s+", label) if p and p[0].isalpha()]
    if not parts:
        return label[:2].upper()
    return (parts[0][0] + (parts[-1][0] if len(parts) > 1 else "")).upper()


def _pair(a: str, b: str) -> tuple[str, str]:
    return tuple(sorted((_norm_id(a), _norm_id(b))))  # type: ignore[return-value]


def _between(payload: dict) -> list[str]:
    """Sorted, normalized endpoint pair as a 2-list (matches build-side edge keying)."""
    return list(_pair(*payload["between"]))


def fold_events(events: list[dict]) -> dict:
    """Fold an append-only correction event-log into the overrides.json contract (pure function).

    Each event: ``{"type": ..., "payload": {...}, "ts": "ISO8601", "editor"?: ..., "note"?: ...}``.
    Last-write-wins per node/paper key (events sorted by ``ts``); add/remove conflicts resolve by
    recency so a later removal can undo an earlier crowd-add. Idempotent: same log -> same dict.
    """
    out = {k: (v.copy() if isinstance(v, (dict, list)) else v) for k, v in EMPTY.items()}
    events = sorted(events, key=lambda e: e.get("ts", ""))  # chronological -> natural LWW

    remove_nodes: set[str] = set()
    remove_edges: dict[tuple, str] = {}      # pair -> latest ts
    add_intent: dict[tuple, tuple] = {}      # (pair, title_norm) -> (ts, item)
    rm_paper: dict[tuple, tuple] = {}        # (pair, title_norm) -> (ts, item)

    for e in events:
        t, p, ts = e.get("type"), e.get("payload", {}), e.get("ts", "")
        if t == "node_label":
            out["node_label"][_norm_id(p["id"])] = p["label"]
        elif t == "node_community":
            out["node_community"][_norm_id(p["id"])] = int(p["community"])
        elif t == "node_url":
            out["node_url"][_norm_id(p["id"])] = {k: p[k] for k in ("openalex", "oa_url") if k in p}
        elif t == "node_photo":
            out["node_photo"][_norm_id(p["id"])] = p["filename"]
        elif t == "paper_rename":
            out["paper_rename"][p["old"]] = p["new"]
        elif t == "remove_node":
            remove_nodes.add(_norm_id(p["id"]))
        elif t == "remove_edge":
            remove_edges[_pair(*p["between"])] = ts
        elif t == "remove_paper":
            rm_paper[(_pair(*p["between"]), _norm_title(p["title"]))] = (
                ts, {"between": _between(p), "title": p["title"]})
        elif t == "add_paper":
            add_intent[(_pair(*p["between"]), _norm_title(p["title"]))] = (
                ts, {"between": _between(p), "title": p["title"], "year": p.get("year")})
        # unknown event types ignored (forward-compatible)

    for (pair, tn), (ts, item) in add_intent.items():
        rp = rm_paper.get((pair, tn))
        if rp and rp[0] > ts:                    # a newer removal cancels this crowd-add
            continue
        if remove_edges.get(pair, "") > ts:      # the whole edge was removed more recently
            continue
        out["add_papers"].append(item)
    for (pair, tn), (ts, item) in rm_paper.items():
        ai = add_intent.get((pair, tn))
        if ai and ai[0] > ts:                    # a newer add re-created it; the add carries it
            continue
        out["remove_papers"].append(item)

    out["remove_edges"] = [list(pair) for pair in remove_edges]
    out["remove_nodes"] = sorted(remove_nodes)
    return out


def load_overrides(path: str | Path) -> dict:
    """Read overrides.json, returning the empty contract if the file is absent."""
    p = Path(path)
    if not p.exists():
        return dict(EMPTY)
    return {**EMPTY, **json.loads(p.read_text())}


def apply_overrides(out: dict, ov: dict) -> dict:
    """Mutate ``out`` (the finished graph dict) in place per ``ov`` and return it."""
    nodes: list[dict] = out["nodes"]
    by_id = {n["id"]: n for n in nodes}
    # index links by unordered endpoint pair (both regular links and fallback path_links)
    norm_id = _norm_id

    # ---- 1. remove_nodes: drop node + every edge/path touching it ------------------------------
    drop = {norm_id(n) for n in ov.get("remove_nodes", [])}
    drop = {nid for nid in drop if nid in {norm_id(k) for k in by_id}}
    if drop:
        dropped_real = {n["id"] for n in nodes if norm_id(n["id"]) in drop}
        out["nodes"] = nodes = [n for n in nodes if n["id"] not in dropped_real]
        by_id = {n["id"]: n for n in nodes}
        out["links"] = [l for l in out["links"]
                        if l["source"] not in dropped_real and l["target"] not in dropped_real]
        out["path_links"] = [l for l in out["path_links"]
                             if l["source"] not in dropped_real and l["target"] not in dropped_real]
        out["paths"] = {u: p for u, p in out["paths"].items()
                        if u not in dropped_real and not (set(p["path"]) & dropped_real)}
        out["unconnected"] = [name for name in out.get("unconnected", [])
                              if norm_id(name) not in drop]
        out["unresolved"] = [name for name in out.get("unresolved", [])
                             if norm_id(name) not in drop]

    # ---- 2. remove_edges: delete the edge outright --------------------------------------------
    rm_edges = {_pair(*e) for e in ov.get("remove_edges", []) if len(e) == 2}
    if rm_edges:
        out["links"] = [l for l in out["links"]
                        if _pair(l["source"], l["target"]) not in rm_edges]

    # ---- 2b. remove_papers: drop a title from an edge (and both endpoints' paper lists) --------
    for item in ov.get("remove_papers", []):
        a, b = item["between"]
        title_key = _norm_title(item["title"])
        pair = _pair(a, b)
        for l in out["links"]:
            if _pair(l["source"], l["target"]) == pair:
                l["papers"] = [t for t in l["papers"] if _norm_title(t) != title_key]
        # also remove from the two endpoints' own paper popups
        for nid in (a, b):
            n = _find(by_id, nid, norm_id)
            if n:
                n["papers"] = [p for p in n["papers"] if _norm_title(p["title"]) != title_key]
    # an edge whose shared-paper list emptied is no longer an edge
    out["links"] = [l for l in out["links"] if l["papers"]]

    # ---- 3. add_papers: find-or-create an edge, append the paper ------------------------------
    for item in ov.get("add_papers", []):
        a, b = item["between"]
        na, nb = _find(by_id, a, norm_id), _find(by_id, b, norm_id)
        if not (na and nb):
            continue  # can't connect a node that isn't in the graph
        title = " ".join(item["title"].split())
        year = item.get("year")
        pair = _pair(a, b)
        link = next((l for l in out["links"] if _pair(l["source"], l["target"]) == pair), None)
        if link is None:
            link = {"source": pair[0], "target": pair[1], "weight": 0,
                    "minhop": 1, "sources": "manual", "papers": []}
            out["links"].append(link)
        if not any(_norm_title(t) == _norm_title(title) for t in link["papers"]):
            link["papers"].append(title)
        for n in (na, nb):
            if not any(_norm_title(p["title"]) == _norm_title(title) for p in n["papers"]):
                n["papers"].append({"title": title, "year": year})

    # ---- 4. paper_rename: rename a title string everywhere it appears -------------------------
    renames = {_norm_title(k): v for k, v in ov.get("paper_rename", {}).items()}
    if renames:
        for n in nodes:
            for p in n["papers"]:
                if _norm_title(p["title"]) in renames:
                    p["title"] = renames[_norm_title(p["title"])]
        for coll in (out["links"], out["path_links"]):
            for l in coll:
                l["papers"] = [renames.get(_norm_title(t), t) for t in l["papers"]]

    # ---- 5. node field patches ----------------------------------------------------------------
    for nid, label in ov.get("node_label", {}).items():
        n = _find(by_id, nid, norm_id)
        if n:
            n["label"], n["initials"] = label, _initials(label)
    comm_ids = {c["id"] for c in out["communities"]}
    for nid, comm in ov.get("node_community", {}).items():
        n = _find(by_id, nid, norm_id)
        if n:
            n["community"] = int(comm)
            if int(comm) not in comm_ids:
                out["communities"].append({"id": int(comm), "label": f"group {comm}"})
                comm_ids.add(int(comm))
    for nid, urls in ov.get("node_url", {}).items():
        n = _find(by_id, nid, norm_id)
        if n:
            if "openalex" in urls:
                n["openalex"] = urls["openalex"]
            if "oa_url" in urls:
                n["oa_url"] = urls["oa_url"]
    for nid, fname in ov.get("node_photo", {}).items():
        n = _find(by_id, nid, norm_id)
        if n:
            n["photo"] = fname if str(fname).startswith("/") else f"{PHOTO_DIR_URL}/{fname}"

    # ---- 6. recompute derived fields + re-assert invariants -----------------------------------
    # Weights stay as built (fractional 1/n_authors per paper — papers.length would stomp them).
    # Only links MINTED by an override (weight 0) get an approximate fractional weight (~1/3 per
    # paper, median team size): the override JSON carries titles, not author counts, so the exact
    # 1/n contribution of a hand-added paper is unknowable here.
    for l in out["links"]:
        if not l["weight"]:
            l["weight"] = round(max(0.2, len(l["papers"]) / 3), 4)
    _recompute_shared_papers(out)
    _recompute_meta(out)
    _assert_invariants(out)
    return out


def _find(by_id: dict, nid: str, norm_id) -> dict | None:
    if nid in by_id:
        return by_id[nid]
    key = norm_id(nid)
    return next((n for n in by_id.values() if norm_id(n["id"]) == key), None)


def _recompute_shared_papers(out: dict) -> None:
    """A node's shared_papers == distinct paper titles across all links touching it (exact)."""
    titles: dict[str, set] = {n["id"]: set() for n in out["nodes"]}
    for l in out["links"]:
        for end in (l["source"], l["target"]):
            if end in titles:
                titles[end].update(_norm_title(t) for t in l["papers"])
    for n in out["nodes"]:
        if not n.get("no_papers") and not n.get("path_only"):
            n["shared_papers"] = len(titles[n["id"]])


def _recompute_meta(out: dict) -> None:
    m = out["meta"]
    m["n_nodes"] = len(out["nodes"])
    m["n_links"] = len(out["links"])
    m["links_by_source"] = dict(Counter(l["sources"] for l in out["links"]))


def _assert_invariants(out: dict) -> None:
    ids = {n["id"] for n in out["nodes"]}
    for l in out["links"]:
        assert l["source"] in ids and l["target"] in ids, f"link endpoint missing: {l}"
        assert l["papers"], f"empty edge survived: {l}"
        assert l["weight"] > 0, f"bad weight: {l}"
    for n in out["nodes"]:
        assert n["x"] == n["x"] and n["y"] == n["y"], f"NaN position: {n['id']}"
    for u, p in out["paths"].items():
        assert all(node in ids for node in p["path"]), f"path references missing node: {u}"
