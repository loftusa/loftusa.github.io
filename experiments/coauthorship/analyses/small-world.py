# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "networkx", "scikit-learn", "graspologic==3.4.4", "setuptools<81"]
# ///
"""small-world — weighted small-world propensity (Muldoon, Bridgeford & Bassett 2016)
of the coauthorship map. Run: cd experiments/coauthorship/analyses && uv run small-world.py"""
import json
import re
import statistics
from pathlib import Path

import networkx as nx
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
GRAPH = json.loads((REPO / "assets/data/coauthorship.json").read_text())
DERIVED = HERE / "_derived"  # unused here; kept per contract template
OUT = REPO / "assets/data/analyses" / "small-world.json"

SLUG = "small-world"
N_RAND = 20  # seeded random-null ensemble size

# ---- build the weighted graph from the shipped links ----------------------------------
node_ids = [n["id"] for n in GRAPH["nodes"]]
community = {n["id"]: n["community"] for n in GRAPH["nodes"]}
label = {n["id"]: n["label"] for n in GRAPH["nodes"]}
is_list = {n["id"]: n["is_list"] for n in GRAPH["nodes"]}
assert len(node_ids) == len(set(node_ids)), "duplicate node ids"

assert GRAPH["meta"]["weighting"] == "fractional", "expected fractional co-authorship weights"
G = nx.Graph()
G.add_nodes_from(node_ids)
pair_papers: dict[frozenset, int] = {}  # display counts stay integer papers, never weights
for l in GRAPH["links"]:
    assert l["source"] in community and l["target"] in community, f"stray endpoint {l}"
    assert 0 < l["weight"] <= l["n_papers"], l  # fractional: each paper adds 1/n_authors
    G.add_edge(l["source"], l["target"], weight=float(l["weight"]))
    pair_papers[frozenset((l["source"], l["target"]))] = int(l["n_papers"])
assert G.number_of_edges() == len(GRAPH["links"]), "parallel links in input?"

comps = sorted(nx.connected_components(G), key=lambda c: (-len(c), sorted(c)[0]))
lcc_nodes = sorted(comps[0])
L_G = G.subgraph(lcc_nodes).copy()
n, m = L_G.number_of_nodes(), L_G.number_of_edges()
assert n >= 90 and m == G.number_of_edges(), "expected all edges inside one big component"
w_max = max(d["weight"] for _, _, d in L_G.edges(data=True))
gu, gv, _ = max(L_G.edges(data=True), key=lambda e: e[2]["weight"])
w_max_papers = pair_papers[frozenset((gu, gv))]
# the method footnote names this pair and count; re-sync the JS prose if this trips
assert {gu, gv} == {"carey e priebe", "hayden helm"} and w_max_papers == 33, (gu, gv, w_max_papers)
weights_multiset = sorted(d["weight"] for _, _, d in L_G.edges(data=True))
assert len(weights_multiset) == m


def mean_weighted_path(graph: nx.Graph) -> float:
    """Mean shortest-path length over unordered pairs, distance = 1/weight."""
    H = graph.copy()
    for u, v, d in H.edges(data=True):
        d["dist"] = 1.0 / d["weight"]
    nodes = sorted(H.nodes)
    tot, cnt = 0.0, 0
    for i, u in enumerate(nodes):
        dist = nx.single_source_dijkstra_path_length(H, u, weight="dist")
        for v in nodes[i + 1 :]:
            assert v in dist, f"graph passed to mean_weighted_path not connected: {u}->{v}"
            tot += dist[v]
            cnt += 1
    assert cnt == len(nodes) * (len(nodes) - 1) // 2
    return tot / cnt


# ---- observed C and L on the weighted LCC ----------------------------------------------
C_obs = nx.average_clustering(L_G, weight="weight")  # Onnela; normalizes by max weight
L_obs = mean_weighted_path(L_G)
assert 0 < C_obs < 1 and L_obs > 0
print(f"[{SLUG}] LCC n={n} m={m} w_max={w_max:.2f} ({w_max_papers} papers)  C={C_obs:.4f}  L={L_obs:.4f}")

# ---- random null: degree-preserving double_edge_swap on the binarized graph, ----------
# ---- then seeded reassignment of the original weight multiset (x N_RAND seeds) --------
C_rand_s, L_rand_s, rand_lcc_sizes = [], [], []
for seed in range(N_RAND):
    B = nx.Graph()
    B.add_nodes_from(lcc_nodes)
    B.add_edges_from(L_G.edges())
    nx.double_edge_swap(B, nswap=10 * m, max_tries=100 * m, seed=seed)
    assert B.number_of_edges() == m
    assert sorted(d for _, d in B.degree()) == sorted(d for _, d in L_G.degree())
    rng = np.random.default_rng(seed)
    perm = rng.permutation(m)
    edges_sorted = sorted(tuple(sorted(e)) for e in B.edges())
    for k, (u, v) in enumerate(edges_sorted):
        B[u][v]["weight"] = weights_multiset[perm[k]]
    C_rand_s.append(nx.average_clustering(B, weight="weight"))
    bcc = max(nx.connected_components(B), key=lambda c: (len(c), sorted(c)[0]))
    rand_lcc_sizes.append(len(bcc))
    L_rand_s.append(mean_weighted_path(B.subgraph(sorted(bcc))))
C_rand = float(np.mean(C_rand_s))
L_rand = float(np.mean(L_rand_s))
rand_lcc_min = min(rand_lcc_sizes)
print(f"[{SLUG}] random null (n={N_RAND}): C={C_rand:.4f} L={L_rand:.4f} lcc_min={rand_lcc_min}")

# ---- lattice null: ring lattice, same n and exactly m edges, ---------------------------
# ---- sorted weights placed nearest-first (deterministic) -------------------------------
lat_edges = []
d_ring = 1
while len(lat_edges) < m:
    for i in range(n):
        if len(lat_edges) >= m:
            break
        j = (i + d_ring) % n
        assert i != j, "ring distance reached n"
        lat_edges.append((i, j))
    d_ring += 1
assert len(lat_edges) == m and len(set(map(frozenset, lat_edges))) == m
LAT = nx.Graph()
LAT.add_nodes_from(range(n))
for k, (i, j) in enumerate(lat_edges):  # nearest-first order; weights descending
    LAT.add_edge(i, j, weight=weights_multiset[m - 1 - k])
assert nx.is_connected(LAT)
C_latt = nx.average_clustering(LAT, weight="weight")
L_latt = mean_weighted_path(LAT)
print(f"[{SLUG}] lattice null: rings to d={d_ring - 1}  C={C_latt:.4f} L={L_latt:.4f}")

# ---- small-world propensity (Muldoon, Bridgeford & Bassett 2016) -----------------------
assert C_latt > C_obs > C_rand, (C_latt, C_obs, C_rand)
assert L_latt > L_obs > L_rand, (L_latt, L_obs, L_rand)
dC_raw = (C_latt - C_obs) / (C_latt - C_rand)
dL_raw = (L_obs - L_rand) / (L_latt - L_rand)
dC = min(1.0, max(0.0, dC_raw))
dL = min(1.0, max(0.0, dL_raw))
swp = 1.0 - ((dC**2 + dL**2) / 2.0) ** 0.5
assert 0.2 < swp < 0.8, f"SWP {swp} outside sane band — re-check before shipping"
print(f"[{SLUG}] dC={dC:.4f} dL={dL:.4f}  SWP={swp:.4f}")

# ---- per-lab subgraph C and L -----------------------------------------------------------
# Onnela clustering normalizes by the max weight of the graph it is given. For a shared
# axis across labs, all three must be normalized by the SAME constant (the map-wide max).
# C  = global normalization (comparable across labs; shown on the shared axis)
# C_own = each lab normalized by its own strongest tie (shipped for honesty: it would
#         rank the Bau lab first — stated in the method footnote)
labs = []
for c in (0, 1, 2):
    members = sorted(v for v in node_ids if community[v] == c)
    assert len(members) >= 5
    sub = G.subgraph(members)
    assert sub.number_of_edges() > 0
    lu, lv, ld = max(
        sub.edges(data=True), key=lambda e: (e[2]["weight"], tuple(sorted((e[0], e[1]))))
    )
    local_max = ld["weight"]
    local_max_papers = pair_papers[frozenset((lu, lv))]
    C_own = nx.average_clustering(sub, weight="weight")
    # global normalization, computed empirically: add a phantom edge carrying the global
    # max weight between two phantom nodes, so networkx normalizes by w_max; the phantom
    # nodes share no triangles with members, then average over members only.
    pad = nx.Graph(sub)
    pad.add_edge("__pad_a__", "__pad_b__", weight=w_max)
    cl = nx.clustering(pad, weight="weight")
    C_glob = sum(cl[v] for v in members) / len(members)
    assert abs(C_glob - C_own * local_max / w_max) < 1e-12  # Onnela is linear in scale
    sub_lcc = max(nx.connected_components(sub), key=lambda s: (len(s), sorted(s)[0]))
    labs.append(
        {
            "id": c,
            "label": GRAPH["communities"][c]["label"],
            "n": len(members),
            "edges": sub.number_of_edges(),
            "C": round(C_glob, 4),
            "C_own": round(C_own, 4),
            "w_max": round(local_max, 4),
            "w_max_papers": local_max_papers,
            "L": round(mean_weighted_path(sub.subgraph(sorted(sub_lcc))), 4),
            "lcc_n": len(sub_lcc),
        }
    )
    assert GRAPH["communities"][c]["id"] == c
tightest = max(labs, key=lambda r: r["C"])
assert tightest["id"] == 2, f"headline assumes Vogelstein lab is tightest; got {tightest}"
print(f"[{SLUG}] labs: {[(r['label'], r['C'], r['C_own'], r['L']) for r in labs]}")

# ---- unweighted hop histogram between list-member pairs --------------------------------
listers = sorted(v for v in lcc_nodes if is_list[v])
hop_counts: dict[int, int] = {}
all_hops = []
for i, u in enumerate(listers):
    dist = nx.single_source_shortest_path_length(L_G, u)
    for v in listers[i + 1 :]:
        h = dist[v]
        hop_counts[h] = hop_counts.get(h, 0) + 1
        all_hops.append(h)
n_pairs = len(listers) * (len(listers) - 1) // 2
assert len(all_hops) == n_pairs
med = statistics.median(all_hops)
assert med == int(med), "even-count median fell between hops; rephrase headline"
med = int(med)
assert med in (2, 3), f"headline phrasing assumes a small median, got {med}"
cum_at_median = sum(1 for h in all_hops if h <= med) / n_pairs
bins = sorted(hop_counts.items())
print(f"[{SLUG}] hops: {bins} median={med} cum@median={cum_at_median:.3f} pairs={n_pairs}")

# ---- the easter egg, verified ----------------------------------------------------------
ERIC = "eric bridgeford"
assert ERIC in community, "Eric Bridgeford missing from GRAPH nodes"
assert community[ERIC] == 2 and is_list[ERIC] and ERIC in lcc_nodes
assert tightest["id"] == community[ERIC]  # his own lab is the tightest-knit

most_word = "most" if cum_at_median >= 0.55 else "half"
headline = (
    f"<strong>{med} handshakes</strong> link {most_word} pairs of list members in the "
    f"map&rsquo;s connected core &mdash; paths nearly as short as pure chance &mdash; "
    f"and the score that says so, small-world propensity <strong>{swp:.2f}</strong>, "
    f"was co-invented by <strong>{label[ERIC]}</strong>: a green dot on this very map."
)

payload = {
    "slug": SLUG,
    "title": "A small world, measured",
    "headline": headline,
    "data": {
        "n": n,
        "m": m,
        "n_total": len(node_ids),
        "w_max": round(w_max, 4),
        "w_max_papers": w_max_papers,
        "C": round(C_obs, 4),
        "L": round(L_obs, 4),
        "C_rand": round(C_rand, 4),
        "L_rand": round(L_rand, 4),
        "C_latt": round(C_latt, 4),
        "L_latt": round(L_latt, 4),
        "dC": round(dC, 4),
        "dL": round(dL, 4),
        "dC_raw": round(dC_raw, 4),
        "dL_raw": round(dL_raw, 4),
        "swp": round(swp, 4),
        "n_rand": N_RAND,
        "rand_lcc_min": rand_lcc_min,
        "lattice_max_ring": d_ring - 1,
        "labs": labs,
        "tightest_lab": tightest["id"],
        "hops": {
            "bins": bins,
            "median": med,
            "cum_at_median": round(cum_at_median, 4),
            "n_pairs": n_pairs,
            "n_members": len(listers),
            "n_list_total": sum(1 for v in node_ids if is_list[v]),
        },
        "eric": {"id": ERIC, "label": label[ERIC], "community": community[ERIC]},
    },
}
assert payload["data"]["eric"]["id"] in community  # every shipped node id exists in GRAPH

blob = json.dumps(payload, separators=(",", ":"))
assert len(blob) < 300_000, f"payload too big: {len(blob)}"
OUT.write_text(blob)
print(f"[{SLUG}] OK {len(blob)/1024:.0f}KB — " + re.sub(r"<[^>]+>", "", headline))
