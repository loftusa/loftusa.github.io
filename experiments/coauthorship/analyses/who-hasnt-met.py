# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "networkx", "scikit-learn", "graspologic==3.4.4", "setuptools<81", "hyppo==0.5.2"]
# ///
"""who-hasnt-met — topic homophily (Dcorr) + missing-edge nominations from title TF-IDF.
Run: cd experiments/coauthorship/analyses && uv run who-hasnt-met.py"""
import json
import math
import re
from pathlib import Path

import networkx as nx
import numpy as np
from hyppo.independence import Dcorr

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
GRAPH = json.loads((REPO / "assets/data/coauthorship.json").read_text())
DERIVED = HERE / "_derived"  # papers.json / yearly.json / layers.json / tfidf.json (read-only)
OUT = REPO / "assets/data/analyses" / "who-hasnt-met.json"

SLUG = "who-hasnt-met"
N_NOMINATIONS = 8
N_TERMS = 3
REPS = 1000

# ---- load inputs ----------------------------------------------------------------------
tfidf = json.loads((DERIVED / "tfidf.json").read_text())
papers = json.loads((DERIVED / "papers.json").read_text())
ids, vocab, rows = tfidf["ids"], tfidf["vocab"], tfidf["rows"]
node_ids = {n["id"] for n in GRAPH["nodes"]}
label = {n["id"]: n["label"] for n in GRAPH["nodes"]}
assert all(i in node_ids for i in ids), "tfidf id missing from shipped graph"
assert len(ids) == len(set(ids)) and len(ids) == len(rows)

# ---- topic vectors -> cosine similarity ------------------------------------------------
M = np.zeros((len(ids), len(vocab)))
for r, row in enumerate(rows):
    for k, w in row:
        M[r, k] = w
norms = np.linalg.norm(M, axis=1)
assert norms.min() > 0.99 and norms.max() < 1.01, "tfidf rows should be ~unit norm"
M = M / norms[:, None]
S = M @ M.T  # cosine similarity, since rows are unit vectors
assert np.allclose(np.diag(S), 1.0) and S.min() >= -1e-9

# ---- graph distance: unweighted shortest-path hops -------------------------------------
G = nx.Graph()
G.add_nodes_from(node_ids)
for l in GRAPH["links"]:
    assert l["source"] in node_ids and l["target"] in node_ids
    G.add_edge(l["source"], l["target"])
assert G.number_of_edges() == len(GRAPH["links"]), "parallel links in input?"

main_cc = max(nx.connected_components(G), key=len)
members = [i for i in ids if i in main_cc]  # tfidf order = alphabetical, deterministic
excluded = [i for i in ids if i not in main_cc]
assert len(members) == 39 and len(excluded) == 5, (len(members), len(excluded))
assert all(G.degree(i) == 0 for i in excluded), "excluded members should be edgeless"
idx = {i: r for r, i in enumerate(ids)}

hops_from = {i: nx.single_source_shortest_path_length(G, i) for i in members}

# papers.json cross-check: every recorded shared paper between two members
co_papers: dict[tuple[str, str], int] = {}
for p in papers:
    ms = [m for m in p["members"] if m in idx]
    for a in range(len(ms)):
        for b in range(a + 1, len(ms)):
            key = tuple(sorted((ms[a], ms[b])))
            co_papers[key] = co_papers.get(key, 0) + 1

# ---- pair table (reachable pairs only; the 6 edgeless members are excluded) ------------
pairs = []  # (id_a, id_b, sim, hops) with a < b in member order
for a in range(len(members)):
    for b in range(a + 1, len(members)):
        i, j = members[a], members[b]
        assert j in hops_from[i], f"unreachable pair inside one component? {i},{j}"
        h = hops_from[i][j]
        assert h >= 1
        pairs.append((i, j, float(S[idx[i], idx[j]]), h))
assert len(pairs) == 39 * 38 // 2 == 741
for i, j, _s, h in pairs[:50]:  # spot-check: hops==1 iff direct edge
    assert (h == 1) == G.has_edge(i, j)

sim_vec = np.array([p[2] for p in pairs])
hop_vec = np.array([p[3] for p in pairs], dtype=float)
assert sim_vec.shape == hop_vec.shape == (741,)

# ---- homophily test: Dcorr(similarity, -hops) ------------------------------------------
res = Dcorr().test(sim_vec, -hop_vec, reps=REPS, auto=False, random_state=0)
dcorr_stat, p_perm = float(res.stat), float(res.pvalue)
pearson_r = float(np.corrcoef(sim_vec, hop_vec)[0, 1])
slope, intercept = (float(v) for v in np.polyfit(sim_vec, hop_vec, 1))
print(f"[{SLUG}] dcorr={dcorr_stat:.4f} p_perm={p_perm:.6f} pearson_r={pearson_r:.4f} "
      f"fit hops = {intercept:.2f} {slope:+.2f}*sim")
assert dcorr_stat > 0.05 and pearson_r < -0.2, "homophily signal vanished — rewrite headline"
assert p_perm < 0.001, "headline claims p < 0.001"

# stricter check: pair rows share people, so also permute PEOPLE (Mantel-style),
# rebuilding the similarity vector each time against the fixed hop vector.
dc = Dcorr()
neg_hop_col = -hop_vec[:, None]  # .statistic() wants 2-D (n, 1), unlike .test()
obs = float(dc.statistic(sim_vec[:, None], neg_hop_col))
assert abs(obs - dcorr_stat) < 1e-9
mem_rows = np.array([idx[i] for i in members])
Smm = S[np.ix_(mem_rows, mem_rows)]
iu = np.triu_indices(len(members), k=1)
assert np.allclose(Smm[iu], sim_vec)
rng = np.random.default_rng(0)
exceed = 0
for _ in range(REPS):
    perm = rng.permutation(len(members))
    if float(dc.statistic(Smm[np.ix_(perm, perm)][iu][:, None], neg_hop_col)) >= obs:
        exceed += 1
p_node = (1 + exceed) / (1 + REPS)
print(f"[{SLUG}] node-permutation (Mantel-style) p={p_node:.6f}")
assert p_node < 0.001, "node-level permutation disagrees — soften the claim"

# ---- nominations: similar topics, no recorded paper together ---------------------------
df = np.count_nonzero(M, axis=0)  # member-level document frequency per term
assert df.shape == (len(vocab),) and df.max() <= len(ids)


def shared_terms(i: str, j: str) -> list[str]:
    """Top distinctive terms both write about: min(weight) * idf, skip numeric tokens."""
    w = np.minimum(M[idx[i]], M[idx[j]]) * np.log(len(ids) / np.maximum(df, 1))
    ranked = sorted(((float(w[k]), vocab[k]) for k in np.nonzero(w)[0]),
                    key=lambda t: (-t[0], t[1]))
    return [t for _v, t in ranked if re.search(r"[a-z]", t)][:N_TERMS]


cand = [p for p in pairs if p[3] >= 2]
skipped = [p for p in cand if co_papers.get((p[0], p[1]), 0) > 0]
assert len(skipped) == 1, f"recheck nomination skip list: {skipped}"  # merullo×brinkmann
cand = [p for p in cand if co_papers.get((p[0], p[1]), 0) == 0]
cand.sort(key=lambda p: (-p[2], p[0], p[1]))
noms = [{"a": i, "b": j, "sim": round(s, 4), "hops": h, "terms": shared_terms(i, j)}
        for i, j, s, h in cand[:N_NOMINATIONS]]
for n in noms:
    assert n["a"] in node_ids and n["b"] in node_ids and len(n["terms"]) >= 2
print(f"[{SLUG}] nominations:")
for n in noms:
    print(f"   {label[n['a']]} × {label[n['b']]}  sim={n['sim']} hops={n['hops']}  both: {n['terms']}")

# headline facts — assert them so a data refresh trips loudly instead of shipping a lie
top = noms[0]
assert (top["a"], top["b"]) == ("jack merullo", "stella biderman"), top
anchor_counts: dict[str, int] = {}
for n in noms:
    for person in (n["a"], n["b"]):
        anchor_counts[person] = anchor_counts.get(person, 0) + 1
anchor, anchor_n = max(anchor_counts.items(), key=lambda kv: (kv[1], kv[0]))
assert (anchor, anchor_n) == ("stella biderman", 6), anchor_counts

headline = (
    f"People who write about the same things really do sit closer on this map "
    f"(distance correlation, <strong>p &lt; 0.001</strong>) &mdash; and where text and "
    f"graph disagree, <strong>{label[anchor]}</strong> anchors {anchor_n} of the "
    f"{len(noms)} collaborations most waiting to happen, starting with {label[top['a']]}."
)

payload = {
    "slug": SLUG,
    "title": "Who hasn't met yet",
    "headline": headline,
    "data": {
        "members": members,
        "excluded": excluded,
        "pairs": [[members.index(i), members.index(j), round(s, 4), h] for i, j, s, h in pairs],
        "test": {
            "dcorr": round(dcorr_stat, 4),
            "p": round(p_perm, 6),
            "reps": REPS,
            "p_node": round(p_node, 6),
            "pearson_r": round(pearson_r, 4),
            "n_pairs": len(pairs),
        },
        "fit": {"slope": round(slope, 3), "intercept": round(intercept, 3)},
        "nominations": noms,
        "nom_cutoff": noms[-1]["sim"],
        "n_skipped_recorded_paper": len(skipped),
    },
}

blob = json.dumps(payload, separators=(",", ":"))
assert len(blob) < 300_000, f"payload too big: {len(blob)}"
OUT.write_text(blob)
print(f"[{SLUG}] OK {len(blob)/1024:.0f}KB — " + re.sub(r"<[^>]+>", "", headline))
