# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "networkx", "scikit-learn", "graspologic==3.4.4", "setuptools<81"]
# ///
"""two-witnesses — do Semantic Scholar and OpenAlex tell the same story about this network?
Latent-position two-sample test (Tang et al. 2017) + per-person cosine agreement.
Edges are fractionally weighted (1/n_authors per pair); cosine runs on those weights, the
global test binarizes, and the disagreement tail is diagnosed on integer paper counts.
Run: cd experiments/coauthorship/analyses && uv run two-witnesses.py"""
import json
import re
import unicodedata
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
GRAPH = json.loads((REPO / "assets/data/coauthorship.json").read_text())
DERIVED = HERE / "_derived"  # layers.json + papers.json (read-only)
OUT = REPO / "assets/data/analyses" / "two-witnesses.json"

SLUG = "two-witnesses"
N_BOOT = 200
TAIL_N = 8

graph_ids = {n["id"] for n in GRAPH["nodes"]}
label = {n["id"]: n["label"] for n in GRAPH["nodes"]}

# ---- build the two witness adjacencies on a shared vertex set -------------------------
# Layer weights are now FRACTIONAL (each paper adds 1/n_authors per pair; suggested by
# Stella Biderman). The per-person cosine (a) runs on these weights; the global test (b)
# binarizes them; the disagreement tail (c) is diagnosed on integer PAPER counts, which we
# re-derive from papers.json below (the layers no longer carry an integer count).
layers = json.loads((DERIVED / "layers.json").read_text())
vo = layers["vertex_order"]
vo_index = {v: i for i, v in enumerate(vo)}
n_all = len(vo)
assert n_all == 119, f"unexpected vertex_order length {n_all}"
assert set(vo) <= graph_ids, "layer vertex not in shipped graph"


def to_matrix(edges: list) -> np.ndarray:
    A = np.zeros((n_all, n_all))
    for i, j, w in edges:
        assert 0 <= i < n_all and 0 <= j < n_all and i != j and w > 0, (i, j, w)
        A[i, j] = w
        A[j, i] = w
    return A


A_s2 = to_matrix(layers["s2"])
A_oa = to_matrix(layers["oa"])
assert np.array_equal(A_s2, A_s2.T) and np.array_equal(A_oa, A_oa.T)

# ---- integer per-source paper counts (for the disagreement tail + biggest disputes) ----
# Re-derive distinct-paper counts per edge per source from papers.json, mirroring the same
# dedup/edge rules build_graph + _prep use, so the integer story and the shipped fractional
# weights come from one and the same clustering.
list_nodes = {n["id"] for n in GRAPH["nodes"] if n["is_list"]}
EDGE_DROP = {tuple(sorted(("jack merullo", "jannik brinkmann")))}


def _clean_display(s: str) -> str:
    if "," in s:
        parts = [p.strip() for p in s.split(",", 1)]
        if len(parts) == 2 and parts[1]:
            s = f"{parts[1]} {parts[0]}"
    return s


def _norm(s: str) -> str:
    s = _clean_display(s)
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower().replace("-", " ")
    return " ".join(re.sub(r"[^\w\s]", " ", s).split())


def _main_title(title: str) -> str:
    head = _norm(title.split(":", 1)[0])
    return head if len(head.split()) >= 3 else ""


def _same_paper(t1: str, t2: str) -> bool:
    n1, n2 = _norm(t1), _norm(t2)
    if not n1 or not n2:
        return False
    if n1 == n2:
        return True
    s1, s2 = set(n1.split()), set(n2.split())
    if s1 <= s2 or s2 <= s1:
        return True
    m1, m2 = _main_title(t1), _main_title(t2)
    return bool(m1) and m1 == m2


def _distinct_count(titles: list) -> int:
    reps: list = []
    for t in titles:
        if not any(_same_paper(t, r) for r in reps):
            reps.append(t)
    return len(reps)


def _pairs_of(p: dict) -> list:
    mem = p["members"] if not p["big"] else [m for m in p["members"] if m in list_nodes]
    out = []
    for a in range(len(mem)):
        for b in range(a + 1, len(mem)):
            e = tuple(sorted((mem[a], mem[b])))
            if e not in EDGE_DROP:
                out.append(e)
    return out


papers = json.loads((DERIVED / "papers.json").read_text())
layer_titles = {"s2": defaultdict(list), "oa": defaultdict(list)}
for p in papers:
    tgt = (["s2"] if "s2" in p["sources"] else []) + (["oa"] if "openalex" in p["sources"] else [])
    for e in _pairs_of(p):
        if e[0] in vo_index and e[1] in vo_index:
            for t in tgt:
                layer_titles[t][e].append(p["title"])


def count_matrix(side: str) -> np.ndarray:
    C = np.zeros((n_all, n_all), dtype=int)
    for (a, b), titles in layer_titles[side].items():
        i, j = vo_index[a], vo_index[b]
        C[i, j] = C[j, i] = _distinct_count(titles)
    return C


C_s2 = count_matrix("s2")
C_oa = count_matrix("oa")
# the integer counts must sit on exactly the shipped fractional edge set — fail loud otherwise
assert np.array_equal(C_s2 > 0, A_s2 > 0), "integer-count topology disagrees with shipped s2 layer"
assert np.array_equal(C_oa > 0, A_oa > 0), "integer-count topology disagrees with shipped oa layer"

# drop vertices with no edge in EITHER witness's account
deg_s2_all = (A_s2 > 0).sum(axis=1)
deg_oa_all = (A_oa > 0).sum(axis=1)
keep = np.where((deg_s2_all > 0) | (deg_oa_all > 0))[0]
kept_ids = [vo[k] for k in keep]
A1 = A_s2[np.ix_(keep, keep)]   # fractional weights
A2 = A_oa[np.ix_(keep, keep)]
K1 = C_s2[np.ix_(keep, keep)]   # integer paper counts (same edge set)
K2 = C_oa[np.ix_(keep, keep)]
n = len(keep)
assert n == 107, f"expected 107 kept vertices, got {n}"
print(f"[{SLUG}] kept {n}/{n_all} vertices; s2 max w {A1.max():.2f}, oa max w {A2.max():.2f}; "
      f"max paper count s2 {K1.max()}, oa {K2.max()}")

# ---- (a) per-person agreement: cosine between each person's two edge vectors ----------
deg1 = (A1 > 0).sum(axis=1)
deg2 = (A2 > 0).sum(axis=1)
people = []   # dual-witness people, with a cosine score
single = []   # people one witness has never seen here
for k in range(n):
    u, v = A1[k], A2[k]
    nu, nv = float(np.linalg.norm(u)), float(np.linalg.norm(v))
    pid = kept_ids[k]
    assert pid in graph_ids
    if nu == 0 or nv == 0:
        assert nu > 0 or nv > 0, "kept vertex isolated in both layers?"
        single.append({"id": pid, "witness": "s2" if nu > 0 else "oa",
                       "deg": int(deg1[k] if nu > 0 else deg2[k])})
    else:
        cos = float(u @ v / (nu * nv))
        assert -1e-9 <= cos <= 1 + 1e-9
        people.append({"id": pid, "cos": round(min(cos, 1.0), 4),
                       "ds2": int(deg1[k]), "doa": int(deg2[k])})

people.sort(key=lambda r: (r["cos"], r["id"]))           # most disputed first
single.sort(key=lambda r: (r["witness"], r["id"]))
assert len(people) == 96 and len(single) == 11, (len(people), len(single))
cos_vals = np.array([r["cos"] for r in people])
median_cos = float(np.median(cos_vals))
frac_above_90 = float((cos_vals > 0.9).mean())
assert median_cos > 0.9, "agreement band collapsed — re-examine"
print(f"[{SLUG}] dual-witness {len(people)}, single-witness {len(single)}; "
      f"median cos {median_cos:.3f}, {frac_above_90:.0%} above 0.9")

# ---- (b) global two-sample latent-position test ----------------------------------------
# The test's parametric bootstrap resamples BINARY Bernoulli graphs from P = X X^T, so the
# observed statistic must live in that same regime. We binarize, and we re-verified the two
# reasons under the new fractional weights:
#   raw fractional weights (max ~5.8): only ~5% of P entries now fall outside [0,1] — far
#     milder than the ~23% the old integer counts produced, but still a weighted graph the
#     binary null cannot match;
#   pass_to_ranks: a smooth (0,1]-weighted observed stat against a BINARY null still puts
#     every one of the 400 null draws above the observed (p = 1.0 by regime mismatch, not by
#     agreement) — exactly the failure the old integer version showed.
# Binarizing also makes this result independent of the weighting change entirely: the
# topology is identical across weightings, so the binarized layers — and this test — are
# unchanged. Edge weights are carried by the per-person cosine above instead.
from graspologic.embed import AdjacencySpectralEmbed  # noqa: E402
from graspologic.inference import latent_position_test  # noqa: E402

np.random.seed(0)
X_raw = AdjacencySpectralEmbed(n_components=3, check_lcc=False).fit_transform(A1)
P_raw = X_raw @ X_raw.T
frac_invalid = float(((P_raw < 0) | (P_raw > 1)).mean())
print(f"[{SLUG}] raw-weight check: {frac_invalid:.0%} of P entries outside [0,1] -> binarize")
# Under fractional weights P sits mostly in-bounds; integer counts would crowd it (~23%).
# A jump back over 0.15 means the weights reverted to integer counts — fail loud.
assert frac_invalid < 0.15, f"raw weights crowd P out of [0,1] ({frac_invalid:.0%}) — weights look integer, not fractional"

B1 = (A1 > 0).astype(float)
B2 = (A2 > 0).astype(float)
n_shared_edges = int((np.triu(B1, 1) * np.triu(B2, 1)).sum())
n_e1 = int(np.triu(B1, 1).sum())
n_e2 = int(np.triu(B2, 1).sum())
print(f"[{SLUG}] edge sets: s2 {n_e1}, oa {n_e2}, shared {n_shared_edges}")

from graspologic.embed import select_dimension  # noqa: E402
d1 = select_dimension(B1)[0][-1]
d2 = select_dimension(B2)[0][-1]
dim = max(d1, d2)
print(f"[{SLUG}] ZG auto dims: s2 {d1}, oa {d2} -> test uses {dim}")

np.random.seed(0)  # latent_position_test draws from the global RNG; workers=1 => sequential
lpt = latent_position_test(B1, B2, embedding="ase", test_case="rotation",
                           n_bootstraps=N_BOOT, workers=1)
stat = float(lpt.stat)
p_value = float(lpt.pvalue)
null1 = np.asarray(lpt.misc_dict["null_distribution_1"])
null2 = np.asarray(lpt.misc_dict["null_distribution_2_"])
assert null1.shape == (N_BOOT,) and null2.shape == (N_BOOT,)
null_min = float(min(null1.min(), null2.min()))
null_max = float(max(null1.max(), null2.max()))
p_floor = 1.0 / (N_BOOT + 1)
print(f"[{SLUG}] LPT stat {stat:.3f}, p {p_value:.4f} (floor {p_floor:.4f}), "
      f"null [{null_min:.3f}, {null_max:.3f}]")
# VERDICT (probed at d=2, 3 and the auto dim — same answer every time): the observed
# difference sits at the BOTTOM of the bootstrap null (stat ≈ its minimum). The two witnesses'
# accounts are statistically indistinguishable — as close as two independent re-samples of the
# same network would be, because both witnessed the same underlying papers (the layers are
# dependent, which makes this test conservative; see prose.method).
assert p_value > 0.95, f"verdict flipped — rewrite headline/prose (p={p_value})"
assert stat < null_min + 0.15 * (null_max - null_min), \
    f"stat no longer near the bottom of the null — rewrite verdict prose (stat={stat}, null=[{null_min},{null_max}])"

# ---- (c) diagnose the disagreement tail FROM the data — on INTEGER paper counts ---------
def neighbors(C: np.ndarray, k: int) -> dict:
    return {kept_ids[j]: int(C[k, j]) for j in range(n) if C[k, j] > 0}


tail = people[:TAIL_N]
assert tail[-1]["cos"] < 0.75 and people[TAIL_N]["cos"] > tail[-1]["cos"]

evidence = {}
for r in tail:
    k = kept_ids.index(r["id"])
    nb1, nb2 = neighbors(K1, k), neighbors(K2, k)
    shared_nb = set(nb1) & set(nb2)
    diffs = sorted(
        ((nb, nb1.get(nb, 0), nb2.get(nb, 0)) for nb in set(nb1) | set(nb2)),
        key=lambda t: (-abs(t[1] - t[2]), t[0]),
    )
    evidence[r["id"]] = {"overlap": len(shared_nb), "n1": len(nb1), "n2": len(nb2),
                         "top_diff": diffs[0]}

# pin the evidence the notes rely on — if the data shifts, fail loudly rather than ship lies
ev = evidence
assert ev["arvind narayanan"]["overlap"] == 0 and ev["arvind narayanan"]["n1"] == 2
assert (ev["jacob steinhardt"]["n1"], ev["jacob steinhardt"]["n2"]) == (5, 9)
assert ev["jacob steinhardt"]["top_diff"] == ("sarah schwettman", 0, 3)
assert ev["c mcdougall"]["overlap"] == 8 and ev["c mcdougall"]["top_diff"] == ("chris wendler", 1, 0)
assert ev["tommy atthey"]["top_diff"] == ("j vogelstein", 6, 1)
assert (ev["goncalo paulo"]["n1"], ev["goncalo paulo"]["n2"]) == (5, 4) and ev["goncalo paulo"]["overlap"] == 2
assert ev["chris wendler"]["top_diff"] == ("robert west", 9, 2)
assert (ev["neel nanda"]["n1"], ev["neel nanda"]["n2"]) == (15, 10)
assert ev["neel nanda"]["top_diff"] == ("clement dumas", 2, 0)
assert (ev["v chandrashekhar"]["n1"], ev["v chandrashekhar"]["n2"]) == (8, 7)
assert ev["v chandrashekhar"]["top_diff"] == ("j vogelstein", 3, 1)

# cause ∈ {coverage gap, name collision, preprint vs published} — always a DATABASE artifact
TAIL_NOTES = {
    "arvind narayanan": ("name collision",
        "The two accounts share zero overlap: Semantic Scholar files a paper with Chris "
        "Wendler and Robert West under this name, OpenAlex the foundation-model policy work "
        "with Percy Liang and Stella Biderman. Two disjoint paper sets under one very common "
        "name — the signature of split or mismatched author profiles."),
    "jacob steinhardt": ("coverage gap",
        "Here OpenAlex is the richer witness: it sees 9 collaborators, including 3 papers "
        "with Sarah Schwettmann and 2 with Kayo Yin, where Semantic Scholar caught only the "
        "5 they both agree on."),
    "c mcdougall": ("coverage gap",
        "Nearly a tie: the witnesses agree paper-for-paper on eight collaborators and split "
        "only on one Semantic-Scholar-only paper, which adds Chris Wendler and a second Neel "
        "Nanda link. A lone few-author paper indexed on one side is enough to separate them."),
    "tommy atthey": ("preprint vs published",
        "Semantic Scholar credits 6 papers to his edge with the “J. Vogelstein” "
        "alias — counting preprint and journal versions the databases never merged — where "
        "OpenAlex matched 1."),
    "goncalo paulo": ("coverage gap",
        "The two witnesses split his EleutherAI papers between them — Semantic Scholar has "
        "the Gleave/McKinney side, OpenAlex the Quirke/Biderman side, and they overlap on "
        "only two collaborators."),
    "chris wendler": ("preprint vs published",
        "The witnesses agree about his Bau-lab collaborations but split 9-vs-2 on his "
        "papers with Robert West: most of that EPFL run is indexed only by Semantic "
        "Scholar — the pattern preprint-heavy work leaves in these databases."),
    "neel nanda": ("coverage gap",
        "Semantic Scholar records 15 collaborators to OpenAlex's 10; the extra five — "
        "Clément Dumas, Julian Minder and three more — all trace to two 2025 preprints "
        "OpenAlex has not yet matched. Where they overlap, they agree."),
    "v chandrashekhar": ("coverage gap",
        "The two agree on his Vogelstein-lab connectomics work except the “J. Vogelstein” "
        "alias edge, where Semantic Scholar counts 3 papers to OpenAlex's 1, plus one extra "
        "link to Eric Bridgeford it alone indexed."),
}
assert set(TAIL_NOTES) == {r["id"] for r in tail}, "tail membership changed — rewrite notes"
for r in tail:
    cause, note = TAIL_NOTES[r["id"]]
    r["cause"] = cause
    r["note"] = note

# ---- the most-disputed edge (headline easter egg) — integer PAPER counts ---------------
pair_w = {}
for i, j in zip(*np.triu_indices(n, k=1)):
    if K1[i, j] > 0 or K2[i, j] > 0:
        pair_w[(kept_ids[i], kept_ids[j])] = (int(K1[i, j]), int(K2[i, j]))
disputes = sorted(pair_w.items(), key=lambda kv: (-abs(kv[1][0] - kv[1][1]), kv[0]))
(top_pair, (tp_s2, tp_oa)) = disputes[0]
assert set(top_pair) == {"carey e priebe", "joshua t vogelstein"}, disputes[:3]
assert (tp_s2, tp_oa) == (20, 27)
# honest tie check: three edges now tie at off-by-7 (priebe×vogelstein, wendler×west,
# vogelstein×ronak mehta); the next gap drops to 6, so the top cluster is unambiguous
assert set(disputes[1][0]) == {"chris wendler", "robert west"}, disputes[:3]
assert disputes[1][1] == (9, 2)
assert abs(disputes[2][1][0] - disputes[2][1][1]) == 7
assert abs(disputes[3][1][0] - disputes[3][1][1]) < 7
print(f"[{SLUG}] most disputed edge: {top_pair} s2={tp_s2} oa={tp_oa}")

# Tang, Athreya, Sussman, Lyzinski & Priebe (JCGS 2017) — Priebe is ON this map.
assert "carey e priebe" in graph_ids and "eric bridgeford" in graph_ids

headline = (
    "Cross-examined, the two databases behind this map tell statistically the same "
    f"story (<strong>p = {p_value:.2f}</strong>, median per-person agreement "
    f"<strong>{median_cos:.2f}</strong>) &mdash; and one of the two biggest bookkeeping "
    "disputes left touches, fittingly, <strong>Carey E. Priebe</strong>, who co-invented "
    "the very test we used to check."
)

payload = {
    "slug": SLUG,
    "title": "Two witnesses",
    "headline": headline,
    "data": {
        "test": {
            "p": round(p_value, 4),
            "p_floor": round(p_floor, 4),
            "stat": round(stat, 3),
            "null_min": round(null_min, 3),
            "null_max": round(null_max, 3),
            "dim": int(dim),
            "n_boot": N_BOOT,
            "n_vertices": n,
            "edges_s2": n_e1,
            "edges_oa": n_e2,
            "edges_shared": n_shared_edges,
        },
        "band": {
            "median": round(median_cos, 4),
            "frac_above_90": round(frac_above_90, 4),
            "n_dual": len(people),
        },
        "people": people,       # ascending cosine: tail first
        "tail_n": TAIL_N,       # first TAIL_N entries of people carry cause + note
        "single": single,
        "dispute": {
            "a": "carey e priebe",
            "b": "joshua t vogelstein",
            "s2": tp_s2,
            "oa": tp_oa,
            "tie": {"a": "chris wendler", "b": "robert west", "s2": 9, "oa": 2},
        },
    },
}

for r in payload["data"]["people"] + payload["data"]["single"]:
    assert r["id"] in graph_ids, r
blob = json.dumps(payload, separators=(",", ":"))
assert len(blob) < 300_000, f"payload too big: {len(blob)}"
OUT.write_text(blob)
print(f"[{SLUG}] OK {len(blob)/1024:.0f}KB — " + re.sub(r"<[^>]+>", "", headline))
