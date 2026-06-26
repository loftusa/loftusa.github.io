# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "networkx", "scikit-learn", "graspologic==3.4.4", "setuptools<81"]
# ///
"""dual-citizens — soft community membership (LSE + GMM responsibilities).
Run: cd experiments/coauthorship/analyses && uv run dual-citizens.py"""
import json
import math
import re
from pathlib import Path

import networkx as nx
import numpy as np
from graspologic.embed import LaplacianSpectralEmbed
from sklearn.mixture import GaussianMixture

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
GRAPH = json.loads((REPO / "assets/data/coauthorship.json").read_text())
OUT = REPO / "assets/data/analyses" / "dual-citizens.json"

SLUG = "dual-citizens"
N_MIXED = 16  # everyone whose runner-up share is >= 1e-11 under fractional weights
              # (rank 17, Percy Liang, drops another factor of ~12 to 9e-13)

# ---- rebuild the site's clustering pipeline (build_graph.py §5), keeping the
#      soft assignments it normally throws away ---------------------------------
assert GRAPH["meta"]["weighting"] == "fractional", "expected fractional edge weights"
nodes = {n["id"]: n for n in GRAPH["nodes"]}
G = nx.Graph()
G.add_nodes_from(nodes)
for l in GRAPH["links"]:
    assert l["source"] in nodes and l["target"] in nodes
    G.add_edge(l["source"], l["target"], weight=l["weight"])

lcc = sorted(max(nx.connected_components(G), key=len))  # same node order as build_graph.py
assert 80 <= len(lcc) < len(nodes), f"unexpected LCC size {len(lcc)}"
A = nx.to_numpy_array(G.subgraph(lcc), nodelist=lcc, weight="weight")
assert A.shape == (len(lcc), len(lcc)) and (A == A.T).all() and A.diagonal().sum() == 0

emb = LaplacianSpectralEmbed(n_components=7, algorithm="full", form="R-DAD").fit_transform(A)  # dim 7 == build_graph.EMBED_DIM; dim 6 stopped separating the anchors once the graph grew
assert emb.shape == (len(lcc), 7) and np.all(np.isfinite(emb))
emb /= np.linalg.norm(emb, axis=1, keepdims=True).clip(1e-9)  # rows onto unit sphere

gmm = GaussianMixture(3, covariance_type="full", random_state=42).fit(emb)
proba = gmm.predict_proba(emb)  # responsibilities = fractional membership
assert proba.shape == (len(lcc), 3) and np.allclose(proba.sum(axis=1), 1.0)

# remap raw GMM components -> fixed community ids via anchor people
ANCHORS = {"stella biderman": 0, "david bau": 1, "j vogelstein": 2}
idx = {nid: i for i, nid in enumerate(lcc)}
assert all(a in idx for a in ANCHORS), "anchor missing from LCC"
raw_of = {a: int(np.argmax(proba[idx[a]])) for a in ANCHORS}
assert len(set(raw_of.values())) == 3, f"anchors not in distinct clusters: {raw_of}"
perm = np.empty(3, dtype=int)  # perm[community id] = raw GMM column
for a, cid in ANCHORS.items():
    perm[cid] = raw_of[a]
proba = proba[:, perm]  # columns now 0=EleutherAI, 1=Bau, 2=Vogelstein
for a, cid in ANCHORS.items():
    assert proba[idx[a]].argmax() == cid, f"anchor {a} lost its column after remap"

# ---- entropy ranking: who is even microscopically mixed? -----------------------
P = proba.clip(1e-300)
entropy = -(P * np.log(P)).sum(axis=1)
order = sorted(range(len(lcc)), key=lambda i: (-entropy[i], lcc[i]))  # deterministic ties

argmax = proba.argmax(axis=1)
top1_share = proba.max(axis=1)
second_share = np.sort(proba, axis=1)[:, 1]
n_pure_9999 = int((top1_share >= 0.9999).sum())
n_measurable = int((second_share >= 1e-4).sum())  # second lab >= 0.01%
n_zero = int((second_share == 0.0).sum())         # runner-up underflows float64
vog_spill = float(proba[argmax != 2, 2].max())    # mass leaking INTO Vogelstein

# two genuine dual citizens have surfaced on the EleutherAI–Bau border (Logan Smith ~98/2,
# Arvind Narayanan ~99.6/0.4); everyone else is still essentially pure. Pin the count so a
# refresh that changes who straddles, or how many, trips loudly.
assert n_pure_9999 == len(lcc) - 2, f"pure-count changed: {n_pure_9999} of {len(lcc)}"
assert n_measurable == 2, f"{n_measurable} measurably-mixed (expected Logan Smith + Arvind Narayanan)"
assert vog_spill < 1e-40, f"Vogelstein frontier is leaking: {vog_spill:.2e}"

# sanity vs shipped colors: at dim 7 the raw model now reproduces Sheridan Feucht's Bau
# placement on its own, but no longer the two COMMUNITY_FORCE recolorings — it reads Kola
# Ayonrinde and Jesse Hoogland as EleutherAI where the site forces them into the Bau lab.
disagree = sorted(nid for nid in lcc if nodes[nid]["community"] != int(argmax[idx[nid]]))
# the expected disagreements are the two deliberate build_graph.COMMUNITY_FORCE overrides
assert disagree == ["jesse hoogland", "kola ayonrinde"], f"unexpected model-vs-site disagreement: {disagree}"
print(f"[{SLUG}] LCC={len(lcc)}  anchors raw->{raw_of}")
print(f"[{SLUG}] pure(>=99.99%)={n_pure_9999}/{len(lcc)}  measurable(>=0.01%)={n_measurable}  "
      f"underflow-to-zero={n_zero}  max spill into Vogelstein={vog_spill:.2e}")
print(f"[{SLUG}] model vs shipped color disagreements: {disagree} (feucht = deliberate override)")

sig3 = lambda x: float(f"{x:.3g}")
mixed_ids = []
for i in order[:N_MIXED]:
    f = proba[i]
    srt = np.argsort(-f, kind="stable")
    assert f[srt[2]] < 1e-6, "third-lab mass not negligible"
    assert {int(srt[0]), int(srt[1])} == {0, 1}, "a trace-mixed person is off the EAI-Bau border"
    mixed_ids.append(lcc[i])
    print(f"  {lcc[i]:28s} top={int(srt[0])} second={int(srt[1])} "
          f"runner-up={f[srt[1]]:.3e}  H={entropy[i]:.2e}")

# per-node record for the minimap + the trace plot (mixed rows look these up by id):
# c = model's argmax lab, j = runner-up lab, s = runner-up share (3 sig figs)
all_nodes = {}
for nid in lcc:
    f = proba[idx[nid]]
    srt = np.argsort(-f, kind="stable")
    all_nodes[nid] = {"c": int(srt[0]), "j": int(srt[1]), "s": sig3(f[srt[1]])}
assert len(all_nodes) == len(lcc) and all(nid in nodes for nid in all_nodes)
assert all(m in all_nodes for m in mixed_ids)

shares = [all_nodes[m]["s"] for m in mixed_ids]
assert all(a > b for a, b in zip(shares, shares[1:])), "trace ranking not strictly ordered"
assert shares[-1] >= 1e-13 and shares[0] < 0.05, f"trace range moved: {shares[0]}..{shares[-1]}"

top_id = mixed_ids[0]
top_label = nodes[top_id]["label"]
top_rec = all_nodes[top_id]                    # {c: top lab, j: runner-up lab, s: runner-up share}
top_pct = int(round(100 * top1_share[idx[top_id]]))
second_id = mixed_ids[1]
second_label = nodes[second_id]["label"]
LAB_NAME = {0: "EleutherAI", 1: "the Bau lab", 2: "the Vogelstein lab"}
# the headline + JS prose name these two; fail loud so they get re-synced if they change
assert top_id == "logan smith", f"most-divided person changed: {top_id}"
assert second_id == "arvind narayanan", f"second dual citizen changed: {second_id}"
assert top_pct == 98, f"headline split changed: {top_pct}"

headline = (
    f"The model has finally started to hesitate: <strong>{top_label}</strong> now splits about "
    f"<strong>{top_pct}/{100 - top_pct}</strong> between {LAB_NAME[top_rec['c']]} and "
    f"{LAB_NAME[top_rec['j']]} &mdash; the map's first real dual citizen since the border snapped "
    f"shut, with <strong>{second_label}</strong> close behind &mdash; while the other "
    f"<strong>{n_pure_9999} of {len(lcc)}</strong> connected people still belong over 99.99% to a "
    f"single lab."
)

payload = {
    "slug": SLUG,
    "title": "Dual citizens",
    "headline": headline,
    "data": {
        "mixed": mixed_ids,
        "nodes": all_nodes,
        "stats": {
            "n_lcc": len(lcc),
            "n_pure_9999": n_pure_9999,
            "n_measurable": n_measurable,
            "n_zero": n_zero,
            "top_id": top_id,
            "vog_spill": sig3(vog_spill),
        },
    },
}
blob = json.dumps(payload, separators=(",", ":"))
assert len(blob) < 300_000, f"payload too big: {len(blob)}"
OUT.write_text(blob)
print(f"[{SLUG}] OK {len(blob)/1024:.0f}KB — " + re.sub(r"<[^>]+>", "", headline))
