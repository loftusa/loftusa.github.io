# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "networkx", "scikit-learn", "graspologic==3.4.4", "setuptools<81"]
# ///
"""stars-and-tribes — the two-truths contrast: ASE sees prominence, LSE sees the labs.
Plus the zoom-out question: is this 2 blocks (interp vs NeuroData) or 3 (the labs)?
Run: cd experiments/coauthorship/analyses && uv run stars-and-tribes.py"""
import json
import re
from pathlib import Path

import networkx as nx
import numpy as np
from graspologic.embed import AdjacencySpectralEmbed, LaplacianSpectralEmbed
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
GRAPH = json.loads((REPO / "public/assets/data/coauthorship.json").read_text())
DERIVED = HERE / "_derived"  # unused here; kept per contract template
OUT = REPO / "public/assets/data/analyses" / "stars-and-tribes.json"

SLUG = "stars-and-tribes"
N_DIMS = 7  # was 6; bumped with build_graph.EMBED_DIM=7 once the graph grew (antonio+eric) — dim 6 dropped LSE-vs-labs ARI below 0.8
# one anchor per lab roster, used to map GMM cluster ids -> community ids
ANCHORS = {"stella biderman": 0, "david bau": 1, "j vogelstein": 2}

# ---- weighted LCC ----------------------------------------------------------------------
community = {n["id"]: n["community"] for n in GRAPH["nodes"]}
label = {n["id"]: n["label"] for n in GRAPH["nodes"]}

assert GRAPH["meta"]["weighting"] == "fractional", "expected fractional edge weights"
G = nx.Graph()
G.add_nodes_from(community)
for l in GRAPH["links"]:
    assert l["source"] in community and l["target"] in community, f"stray endpoint {l}"
    assert l["weight"] > 0 and int(l["n_papers"]) >= 1
    G.add_edge(l["source"], l["target"], weight=float(l["weight"]), n_papers=int(l["n_papers"]))
assert G.number_of_edges() == len(GRAPH["links"]), "parallel links in input?"

lcc_ids = sorted(max(nx.connected_components(G), key=len))
n = len(lcc_ids)
assert n == 107, f"unexpected LCC size {n}"
assert all(community[v] in (0, 1, 2) for v in lcc_ids), "LCC node without a lab community"
assert set(lcc_ids) == {v for v, c in community.items() if c in (0, 1, 2)}, \
    "LCC should be exactly the three lab communities"
assert all(a in lcc_ids for a in ANCHORS), "anchor missing from LCC"

A = nx.to_numpy_array(G.subgraph(lcc_ids), nodelist=lcc_ids, weight="weight")
assert A.shape == (n, n) and np.allclose(A, A.T) and A.min() >= 0
wdeg = A.sum(axis=1)
assert wdeg.min() > 0, "isolated vertex inside the LCC?"
site = np.array([community[v] for v in lcc_ids])

# ---- ASE: adjacency spectral embedding — its first axis is prominence -------------------
ase = AdjacencySpectralEmbed(n_components=N_DIMS, algorithm="full").fit_transform(A)
assert ase.shape == (n, N_DIMS)
if np.corrcoef(ase[:, 0], wdeg)[0, 1] < 0:  # orient dim 1 positive with weighted degree
    ase[:, 0] *= -1
for d in range(1, N_DIMS):  # fix remaining sign ambiguity deterministically
    if ase[np.argmax(np.abs(ase[:, d])), d] < 0:
        ase[:, d] *= -1
prominence = ase[:, 0]
assert prominence.min() > -1e-9, "first ASE axis of a connected nonneg matrix should be >= 0"
r_deg = float(np.corrcoef(prominence, wdeg)[0, 1])
assert abs(r_deg) > 0.5, f"prominence–degree correlation suspiciously weak: {r_deg}"

# ---- LSE: degree-normalized Laplacian embedding — it sees the tribes --------------------
lse = LaplacianSpectralEmbed(n_components=N_DIMS, form="R-DAD").fit_transform(A)
assert lse.shape == (n, N_DIMS)
norms = np.linalg.norm(lse, axis=1, keepdims=True)
assert norms.min() > 1e-12
lse = lse / norms  # row-normalize (project to the sphere; standard before GMM)
for d in range(N_DIMS):
    if lse[np.argmax(np.abs(lse[:, d])), d] < 0:
        lse[:, d] *= -1

gmm_raw = GaussianMixture(n_components=3, random_state=42).fit_predict(lse)
ari_lse = float(adjusted_rand_score(site, gmm_raw))
assert ari_lse > 0.8, f"LSE clusters should recover the labs; ARI={ari_lse}"

# anchor-remap GMM cluster ids onto community ids
idx = {v: i for i, v in enumerate(lcc_ids)}
remap = {int(gmm_raw[idx[a]]): c for a, c in ANCHORS.items()}
assert len(remap) == 3, f"anchors landed in only {len(remap)} GMM clusters: {remap}"
gmm = np.array([remap[int(g)] for g in gmm_raw])
n_disagree = int((gmm != site).sum())
assert n_disagree <= 5, f"too many roster/cluster disagreements: {n_disagree}"

# contrast: the SAME GMM on the ASE embedding does not find the labs
ari_ase = float(adjusted_rand_score(site, GaussianMixture(3, random_state=42).fit_predict(ase)))
assert ari_ase < 0.5, f"ASE clusters unexpectedly lab-like: {ari_ase}"

# ---- best 2 separating LSE dims (one-way ANOVA F ratio against the lab rosters) ---------
def f_ratio(col: np.ndarray) -> float:
    mu = col.mean()
    between = sum((col[site == c].mean() - mu) ** 2 * (site == c).sum() for c in (0, 1, 2))
    within = sum(((col[site == c] - col[site == c].mean()) ** 2).sum() for c in (0, 1, 2))
    return between / max(within, 1e-12)

f_by_dim = [f_ratio(lse[:, d]) for d in range(N_DIMS)]
lse_dims = sorted(sorted(range(N_DIMS), key=lambda d: -f_by_dim[d])[:2])
print(f"[{SLUG}] LSE F-ratios per dim: {[round(f, 2) for f in f_by_dim]} -> dims {lse_dims}")

# ---- the 6 names labeled in BOTH clouds: 3 anchors + 3 highest weighted degree ----------
wd_of = dict(zip(lcc_ids, wdeg))
# skip the second database identity of an anchor (e.g. "joshua t vogelstein" duplicates
# "j vogelstein") so we never label the same human twice
dup_of_anchor = {"joshua t vogelstein"}
high = sorted(
    (v for v in lcc_ids if v not in ANCHORS and v not in dup_of_anchor),
    key=lambda v: (-wd_of[v], v),
)[:3]
labeled = sorted(ANCHORS) + high
assert len(set(labeled)) == 6
assert "carey e priebe" in labeled, "expected Priebe among the high-degree labels"

top_star = max(lcc_ids, key=lambda v: prominence[idx[v]])
print(f"[{SLUG}] top prominence: {top_star} ({label[top_star]}), labeled: {labeled}")
print(f"[{SLUG}] r(prominence, wdeg)={r_deg:.4f}  ARI(LSE,labs)={ari_lse:.4f}  "
      f"ARI(ASE,labs)={ari_ase:.4f}  disagreements={n_disagree}")

# ---- zoom out: 2 blocks (interp vs NeuroData) or 3 (the labs)? --------------------------
# Alex's reading: EleutherAI+Bau are ONE core-periphery interp block, Vogelstein separate.
# Test: degree-corrected SBM with memberships FIXED to k=2 {interp, Vogelstein} vs k=3
# (the labs); Karrer–Newman (2011) Poisson profile log-likelihood on the fractional
# weights (a quasi-likelihood — fractional weights are not counts), compared by BIC with
# M = n(n-1)/2 dyads and p = k(k+1)/2 block params + (n-k) free degree params.
def kn_profile_loglik(part: np.ndarray) -> float:
    ks = sorted(set(part))
    m = np.array([[A[np.ix_(part == a, part == b)].sum() for b in ks] for a in ks])
    assert np.allclose(m, m.T) and m.sum() > 0
    kappa = m.sum(axis=1)
    return float(sum(m[a, b] * np.log(m[a, b] / (kappa[a] * kappa[b]))
                     for a in range(len(ks)) for b in range(len(ks)) if m[a, b] > 0))

part3 = site
part2 = np.where(site == 2, 1, 0)  # EleutherAI+Bau -> block 0, Vogelstein -> block 1
ll3, ll2 = kn_profile_loglik(part3), kn_profile_loglik(part2)
assert ll3 >= ll2 - 1e-9, "k=2 is a coarsening of k=3; its likelihood cannot win"
M = n * (n - 1) / 2
bic = lambda ll, k: -2 * ll + (k * (k + 1) / 2 + n - k) * np.log(M)
bic2, bic3 = bic(ll2, 2), bic(ll3, 3)
dbic = float(bic2 - bic3)  # positive => the 3-block labs beat {interp, NeuroData}
assert dbic > 10, f"prose says k=3 wins decisively; re-sync if this flips (dbic={dbic})"

# supporting seam stats (block ids: 0 EleutherAI, 1 Bau, 2 Vogelstein)
def seam(a: int, b: int):
    edges = [(u, v, d) for u, v, d in G.edges(data=True)
             if {community[u], community[v]} == {a, b}]
    return (len(edges), int(sum(d["n_papers"] for _, _, d in edges)),
            float(sum(d["weight"] for _, _, d in edges)))

def block_w(c: int) -> float:
    return float(sum(d["weight"] for u, v, d in G.edges(data=True)
                     if community[u] == community[v] == c))

ev_edges, _, _ = seam(0, 2)
assert ev_edges == 0, f"prose says ZERO direct EleutherAI–Vogelstein edges; found {ev_edges}"
eb_edges, eb_papers, eb_w = seam(0, 1)
bv_edges, bv_papers, bv_w = seam(1, 2)
eb_share = eb_w / (block_w(0) + block_w(1) + eb_w)      # seam share of interp-block weight
bv_share = bv_w / (block_w(1) + block_w(2) + bv_w)      # seam share of Bau+Vogelstein weight
assert 0.02 < eb_share < 0.5 and bv_share < 0.05, f"seam shares odd: {eb_share}, {bv_share}"

# interp-internal core-periphery gradient: weighted degree vs ASE radius (L2 norm, d=6)
interp = np.isin(site, [0, 1])
assert int(interp.sum()) == int((site == 0).sum() + (site == 1).sum())
r_core = float(np.corrcoef(wdeg[interp], np.linalg.norm(ase, axis=1)[interp])[0, 1])
assert r_core > 0.3, f"core-periphery gradient unexpectedly weak: {r_core}"

print(f"[{SLUG}] DCSBM: logL k3={ll3:.2f} k2={ll2:.2f}  BIC k3={bic3:.1f} k2={bic2:.1f}  "
      f"dBIC={dbic:.1f} (k=3 wins)")
print(f"[{SLUG}] seams: E<->V edges={ev_edges}  E<->B {eb_edges} edges/{eb_papers} papers "
      f"share={eb_share:.3f}  B<->V {bv_edges} edges/{bv_papers} papers share={bv_share:.4f}  "
      f"r_core={r_core:.3f} (n_interp={int(interp.sum())})")

# ---- ship -------------------------------------------------------------------------------
sig = lambda x: float(f"{x:.4g}")
nodes_out = []
for v in lcc_ids:
    i = idx[v]
    assert v in community, f"shipped id {v} not in GRAPH nodes"
    nodes_out.append({
        "id": v,
        "ase": [sig(ase[i, 0]), sig(ase[i, 1])],
        "lse": [sig(lse[i, lse_dims[0]]), sig(lse[i, lse_dims[1]])],
        "c": int(site[i]),          # roster lab
        "g": int(gmm[i]),           # what the Laplacian clustering says
        "wd": sig(wd_of[v]),        # fractional weighted degree — NOT a paper count
    })

# NOTE: the map's community colors were themselves assigned by an LSE+GMM recipe (see
# experiments/coauthorship/README.md §4), so ARI(LSE, site colors) is self-consistency,
# not external validation — the headline says "re-deriving the map's colors", and the
# discriminating fact is that ASE scores ~0 on the identical target. Under fractional
# weights the only model-vs-site difference is the deliberate Feucht editorial override
# (overrides.json: a Bau-lab student the raw model reads as EleutherAI), so the count
# is derived from n_disagree rather than hardcoded.
match_str = f"all {n} of {n}" if n_disagree == 0 else f"{n - n_disagree} of {n}"
headline = (
    "Same graph, two honest readings: the adjacency embedding&rsquo;s first axis is sheer "
    f"prominence &mdash; r&nbsp;=&nbsp;<strong>{r_deg:.2f}</strong> with weighted degree, "
    f"<strong>{label[top_star]}</strong> out front &mdash; while the degree-normalized "
    "Laplacian sees tribes instead, re-deriving the map&rsquo;s three lab colors for "
    f"<strong>{match_str}</strong> people."
)

payload = {
    "slug": SLUG,
    "title": "Stars and tribes",
    "headline": headline,
    "data": {
        "stats": {
            "r_deg": sig(r_deg),
            "ari_lse": sig(ari_lse),
            "ari_ase": sig(ari_ase),
            "n_lcc": n,
            "n_disagree": n_disagree,
            "lse_dims": lse_dims,
            "top_star": top_star,
        },
        "seam": {                       # the 2-block vs 3-block verdict + supporting stats
            "dbic": sig(dbic),          # BIC(k=2) - BIC(k=3); positive => 3 labs win
            "ev_edges": ev_edges,       # direct EleutherAI<->Vogelstein edges (zero)
            "eb_edges": eb_edges, "eb_papers": eb_papers, "eb_share": sig(eb_share),
            "bv_edges": bv_edges, "bv_papers": bv_papers, "bv_share": sig(bv_share),
            "r_core": sig(r_core),      # interp-internal corr(weighted degree, ASE radius)
            "n_interp": int(interp.sum()),
        },
        "anchors": sorted(ANCHORS),
        "labeled": labeled,
        "nodes": nodes_out,
    },
}

blob = json.dumps(payload, separators=(",", ":"))
assert len(blob) < 300_000, f"payload too big: {len(blob)}"
OUT.write_text(blob)
print(f"[{SLUG}] OK {len(blob)/1024:.0f}KB — " + re.sub(r"<[^>]+>", "", headline))
