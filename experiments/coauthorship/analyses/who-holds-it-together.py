# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "networkx", "scikit-learn", "graspologic==3.4.4", "setuptools<81"]
# ///
"""who-holds-it-together — mediator ablation: remove each person, measure lost between-lab
max-flow. Run: cd experiments/coauthorship/analyses && uv run who-holds-it-together.py"""
import json
import re
from pathlib import Path

import networkx as nx
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
GRAPH = json.loads((REPO / "public/assets/data/coauthorship.json").read_text())
DERIVED = HERE / "_derived"  # unused here; kept per contract template
OUT = REPO / "public/assets/data/analyses" / "who-holds-it-together.json"

SLUG = "who-holds-it-together"
PAIRS = [(0, 1), (0, 2), (1, 2)]
TOP_N = 14

# ---- build the weighted graph from the shipped links --------------------------------
# weights are FRACTIONAL co-authorship counts (each paper adds 1/n_authors per pair),
# so flows are tie-strength units, not paper counts. Fail fast if the input regresses.
assert GRAPH["meta"]["weighting"] == "fractional", GRAPH["meta"].get("weighting")
node_ids = [n["id"] for n in GRAPH["nodes"]]
community = {n["id"]: n["community"] for n in GRAPH["nodes"]}
degree = {n["id"]: n["degree"] for n in GRAPH["nodes"]}
label = {n["id"]: n["label"] for n in GRAPH["nodes"]}
assert len(node_ids) == len(set(node_ids)), "duplicate node ids"

G = nx.Graph()
G.add_nodes_from(node_ids)
for l in GRAPH["links"]:
    assert l["source"] in community and l["target"] in community, f"stray endpoint {l}"
    assert l["weight"] > 0
    G.add_edge(l["source"], l["target"], capacity=float(l["weight"]))
assert G.number_of_edges() == len(GRAPH["links"]), "parallel links in input?"

members = {c: sorted(v for v in node_ids if community[v] == c) for c in (0, 1, 2)}
for c in (0, 1, 2):
    assert len(members[c]) >= 5, f"lab {c} suspiciously small"


def pair_flow(graph: nx.Graph, a: int, b: int) -> float:
    """Max-flow between lab a and lab b with each lab contracted to one supernode.

    Contraction sums parallel capacities; intra-lab edges vanish (a lab's internal
    capacity is not the bottleneck under study). Third-lab members and unaffiliated
    nodes stay as individual carriers.
    """
    sa, sb = f"__LAB{a}__", f"__LAB{b}__"
    side = {}
    for v in graph.nodes:
        if community.get(v) == a:
            side[v] = sa
        elif community.get(v) == b:
            side[v] = sb
        else:
            side[v] = v
    H = nx.Graph()
    H.add_nodes_from([sa, sb])    # list, not set: node insertion order must be run-stable
    for u, v, d in graph.edges(data=True):
        mu, mv = side[u], side[v]
        if mu == mv:
            continue
        w = d["capacity"] + (H[mu][mv]["capacity"] if H.has_edge(mu, mv) else 0.0)
        H.add_edge(mu, mv, capacity=w)
    if not (H.has_node(sa) and H.has_node(sb)):
        return 0.0
    return float(nx.maximum_flow(H, sa, sb, capacity="capacity")[0])


def lambda2_lcc(graph: nx.Graph) -> float:
    """Algebraic connectivity (Fiedler value) of the largest connected component."""
    lcc = max(nx.connected_components(graph), key=lambda c: (len(c), sorted(c)[0]))
    sub = graph.subgraph(sorted(lcc))
    L = nx.laplacian_matrix(sub, nodelist=sorted(lcc), weight="capacity").toarray()
    eig = np.linalg.eigvalsh(L.astype(np.float64))
    assert eig[0] < 1e-8, "Laplacian smallest eigenvalue should be ~0"
    return float(eig[1])


# ---- baseline ------------------------------------------------------------------------
base_flows = [pair_flow(G, a, b) for a, b in PAIRS]
base_total = sum(base_flows)
assert base_total > 0, "no between-lab flow at all?"
base_l2 = lambda2_lcc(G)
print(f"[{SLUG}] baseline flows {base_flows} total {base_total} lambda2 {base_l2:.4f}")

# sanity: direct cross-lab edge weight per pair (for the 'no direct papers' claim)
direct = {p: 0.0 for p in PAIRS}
for u, v, d in G.edges(data=True):
    cu, cv = community[u], community[v]
    if cu >= 0 and cv >= 0 and cu != cv:
        direct[tuple(sorted((cu, cv)))] += d["capacity"]
print(f"[{SLUG}] direct cross-lab weights {direct}")
# prose claim: EleutherAI and Vogelstein share no papers directly (zero 0-2 edges)
assert direct[(0, 2)] == 0.0, direct

# ---- ablate every node ----------------------------------------------------------------
rows = []
for v in node_ids:
    Gv = G.copy()
    Gv.remove_node(v)
    flows_v = [pair_flow(Gv, a, b) for a, b in PAIRS]
    for f0, f1 in zip(base_flows, flows_v):
        assert f1 <= f0 + 1e-9, f"removal increased flow for {v}: {f0} -> {f1}"
    cost = 1.0 - sum(flows_v) / base_total
    per_pair = [
        (1.0 - f1 / f0) if f0 > 0 else 0.0 for f0, f1 in zip(base_flows, flows_v)
    ]
    assert -1e-9 <= cost <= 1.0 + 1e-9
    rows.append({"id": v, "cost": cost, "per_pair": per_pair})

def r4(x: float) -> float:
    """Round to 4 decimals and normalize -0.0 -> 0.0 (max-flow float wobble at ~1e-17
    otherwise flips the sign bit between runs and breaks byte-determinism)."""
    return round(x, 4) + 0.0


# deterministic ranking: ROUNDED cost desc (collapses float wobble), then degree desc, then id
rows.sort(key=lambda r: (-r4(r["cost"]), -degree[r["id"]], r["id"]))

# collapse database name-twins (same human, split profiles — e.g. "samuel marks" /
# "samuel d marks"): keep the higher-ranked variant so one person isn't listed twice.
seen_person, deduped = set(), []
for r in rows:
    toks = r["id"].split()
    person = (toks[0], toks[-1])
    if person in seen_person:
        continue
    seen_person.add(person)
    deduped.append(r)
top = deduped[:TOP_N]
assert top[0]["cost"] > 0.05, "top remover should matter"

# secondary check (only for the shipped people): drop in algebraic connectivity
for r in top:
    Gv = G.copy()
    Gv.remove_node(r["id"])
    r["ac_drop"] = r4(1.0 - lambda2_lcc(Gv) / base_l2)
    r["degree"] = degree[r["id"]]
    r["cost"] = r4(r["cost"])
    r["per_pair"] = [r4(x) for x in r["per_pair"]]
    assert r["id"] in community, f"shipped id {r['id']} not in GRAPH nodes"

# ---- verify the two headline facts -----------------------------------------------------
top1 = top[0]
# the #1 mediator by share of lab-to-lab flow is now Kola Ayonrinde — shifted from Sheridan
# Feucht as antonio + eric thickened the Bau community around her bridging edges
assert top1["id"] == "kola ayonrinde", f"unexpected #1: {top1['id']}"  # update headline if this trips
severers = [r for r in top if r["per_pair"][1] > 0.999 and r["per_pair"][2] > 0.999]
assert len(severers) == 1 and severers[0]["id"] == "alex loftus", severers
# independent structural check: every edge leaving community 2 must touch alex loftus
# (he now sits INSIDE community 2 after the re-clustering; all its outward edges are his)
boundary = [
    (u, v)
    for u, v in G.edges
    if (community[u] == 2) != (community[v] == 2)
]
assert all("alex loftus" in e for e in boundary), boundary
print(f"[{SLUG}] top5: {[(r['id'], r['cost']) for r in top[:5]]}")

headline = (
    f"Remove <strong>{label[top1['id']]}</strong> and <strong>"
    f"{round(100 * top1['cost'])}%</strong> of all lab-to-lab flow disappears &mdash; "
    f"and <strong>{label['alex loftus']}</strong> is single-handedly the Vogelstein "
    f"lab&rsquo;s entire link to everyone else."
)

payload = {
    "slug": SLUG,
    "title": "Who holds it together",
    "headline": headline,
    "data": {
        "pairs": PAIRS,
        "baseline": {
            # tie-strength units (fractional co-authorship), rounded to kill float noise
            "flows": [r4(f) for f in base_flows],
            "total": r4(base_total),
            "direct_02": r4(direct[(0, 2)]),
            "lambda2": round(base_l2, 4),
        },
        "people": top,
    },
}

blob = json.dumps(payload, separators=(",", ":"))
assert len(blob) < 300_000, f"payload too big: {len(blob)}"
OUT.write_text(blob)
print(f"[{SLUG}] OK {len(blob)/1024:.0f}KB — " + re.sub(r"<[^>]+>", "", headline))
