# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "networkx", "scikit-learn", "graspologic==3.4.4", "setuptools<81"]
# ///
"""heating-up — vertex-centered scan statistic (Priebe et al. 2005) on the yearly cumulative
coauthorship graph. Run: cd experiments/coauthorship/analyses && uv run heating-up.py"""
import json
import re
import unicodedata
from pathlib import Path

import numpy as np


# ---- title-dedup helpers, vendored from build_graph.py (KEEP IN SYNC) -------------------
def _norm_t(s: str) -> str:
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower().replace("-", " ")
    return " ".join(re.sub(r"[^\w\s]", " ", s).split())


def _main_t(t: str) -> str:
    h = _norm_t(t.split(":", 1)[0])
    return h if len(h.split()) >= 3 else ""


def _same_paper(t1: str, t2: str) -> bool:
    n1, n2 = _norm_t(t1), _norm_t(t2)
    if not n1 or not n2:
        return False
    if n1 == n2:
        return True
    s1, s2 = set(n1.split()), set(n2.split())
    if s1 <= s2 or s2 <= s1:
        return True
    m1, m2 = _main_t(t1), _main_t(t2)
    return bool(m1) and m1 == m2


def distinct_count(titles: list[str]) -> int:
    """# distinct papers, collapsing version/variant duplicate titles."""
    reps: list[str] = []
    for t in titles:
        if not any(_same_paper(t, r) for r in reps):
            reps.append(t)
    return len(reps)

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
GRAPH = json.loads((REPO / "public/assets/data/coauthorship.json").read_text())
DERIVED = HERE / "_derived"
OUT = REPO / "public/assets/data/analyses" / "heating-up.json"

SLUG = "heating-up"
SCAN_YEARS = (2024, 2025, 2026)  # the "recent" window the scan maximizes over
N_RUNNERS = 5
# No randomness anywhere in this script — output is deterministic by construction.

# Weights are FRACTIONAL co-authorship counts (each paper adds 1/n_authors per pair),
# so the locality statistic is in "collaboration weight" units, not pair×paper counts.
assert GRAPH["meta"]["weighting"] == "fractional", "expected fractional edge weighting"

# ------------------------------------------------------------------- load yearly graphs
yearly = json.loads((DERIVED / "yearly.json").read_text())
V: list[str] = yearly["vertex_order"]
YEARS: list[int] = yearly["years"]
n, T = len(V), len(YEARS)
label_of = {nd["id"]: nd["label"] for nd in GRAPH["nodes"]}
assert set(V) <= set(label_of), "yearly vertex_order must be a subset of shipped graph ids"
assert all(y in YEARS for y in SCAN_YEARS) and YEARS == sorted(YEARS)

A = np.zeros((T, n, n))  # cumulative weighted adjacency per year
for ti, yr in enumerate(YEARS):
    for i, j, w in yearly["cumulative"][str(yr)]:
        assert i != j and w > 0
        A[ti, i, j] = w
        A[ti, j, i] = w
assert np.all(np.diff(A, axis=0) >= 0), "cumulative graphs must be monotone"

# ------------------- locality statistic W_t(v)  (the weighted Psi_{t;1}(v) of the paper)
# W[t, v] = total edge weight of the subgraph induced by v's closed 1-hop neighborhood
W = np.zeros((T, n))
for ti in range(T):
    a = A[ti]
    for v in range(n):
        nbrs = np.flatnonzero(a[v])
        if nbrs.size:
            idx = np.r_[v, nbrs]
            W[ti, v] = a[np.ix_(idx, idx)].sum() / 2.0
assert np.isfinite(W).all() and np.all(W >= 0)
assert not np.allclose(W, np.round(W)), "W looks integral — expected fractional weights"
assert np.all(np.diff(W, axis=0) >= 0), "locality stat on cumulative graphs is monotone"

# independent dict-based recomputation for every scan year (no numpy indexing tricks)
for yr in SCAN_YEARS:
    adj: dict[int, dict[int, float]] = {}
    for i, j, w in yearly["cumulative"][str(yr)]:
        adj.setdefault(i, {})[j] = w
        adj.setdefault(j, {})[i] = w
    ti = YEARS.index(yr)
    for v in range(n):
        nb = set(adj.get(v, {})) | {v}
        ww = sum(w for i in nb for j, w in adj.get(i, {}).items() if j in nb) / 2.0
        assert abs(ww - W[ti, v]) < 1e-9, f"cross-check failed: {V[v]} @ {yr}"

# --------------------------------------------------- standardized scan statistic S_t(v)
# D_t = W_t - W_{t-1}; standardize against v's OWN past differences (sd floored at 1).
# Debut rule: if v had no neighborhood at t-1 (W_{t-1} == 0) there is no baseline of its
# own to exceed — skip. (Also keeps name-split debut records from topping the list.)
D = np.diff(W, axis=0)  # D[k, v] = the jump during year YEARS[k+1]
best: dict[int, tuple[float, int]] = {}  # v -> (best S, year index)
for ti, yr in enumerate(YEARS):
    if yr not in SCAN_YEARS:
        continue
    past = D[: ti - 1]  # jumps for years YEARS[1] .. YEARS[ti-1]
    assert past.shape[0] >= 2
    mu, sd = past.mean(axis=0), np.maximum(past.std(axis=0), 1.0)
    S = (D[ti - 1] - mu) / sd
    for v in range(n):
        if W[ti - 1, v] <= 0:
            continue
        if v not in best or S[v] > best[v][0]:
            best[v] = (float(S[v]), ti)

ranked = sorted(best.items(), key=lambda kv: (-kv[1][0], kv[0]))  # deterministic ties
assert len(ranked) > N_RUNNERS

# ------------------------------------------------------------------- the hot neighborhood
hv, (hot_S, hti) = ranked[0]
center, hot_year = V[hv], YEARS[hti]
members = sorted({V[i] for i in np.flatnonzero(A[hti, hv])} | {center})
assert center in members and len(members) >= 3
assert all(m in label_of for m in members)

series = [[yr, round(float(W[ti, hv]), 2)] for ti, yr in enumerate(YEARS)]
w_prev, w_hot = round(float(W[hti - 1, hv]), 2), round(float(W[hti, hv]), 2)
d_hot = round(float(W[hti, hv] - W[hti - 1, hv]), 2)
assert w_prev > 0 and d_hot > 0

# human-readable anchor: DISTINCT papers (version-variants collapsed) in the hot year
# with >=2 of the members on them
papers = json.loads((DERIVED / "papers.json").read_text())
mset = set(members)
papers_hot = distinct_count([p["title"] for p in papers if isinstance(p.get("year"), int)
                             and p["year"] == hot_year and len(mset & set(p["members"])) >= 2])
assert papers_hot >= 5, f"hot-year paper count suspiciously low: {papers_hot}"

# pre-hot linear trend, extrapolated to the latest year for the "+N above trend" bracket
pre_x = np.array([yr for yr in YEARS if yr < hot_year], dtype=float)
pre_y = np.array([W[ti, hv] for ti, yr in enumerate(YEARS) if yr < hot_year], dtype=float)
slope, intercept = np.polyfit(pre_x, pre_y, 1)
gap_last = float(W[-1, hv] - (slope * YEARS[-1] + intercept))
assert gap_last > 0

runners = [
    {"id": V[v], "S": round(S, 1), "year": YEARS[ti],
     "w_prev": round(float(W[ti - 1, v]), 2), "w": round(float(W[ti, v]), 2)}
    for v, (S, ti) in ranked[1 : 1 + N_RUNNERS]
]
assert all(r["id"] in label_of for r in runners)

# -------------------------------------------------- easter egg: scan the scan's inventor
# prose.method hardcodes the verdict ("dead last — 95th of the 95 scored") — keep it honest.
pv = V.index("carey e priebe")
p_rank = 1 + next(k for k, (v, _) in enumerate(ranked) if v == pv)
assert (p_rank, len(ranked)) == (101, 102), \
    f"Priebe verdict changed ({p_rank}/{len(ranked)}) — update prose.method in heating-up.js"
assert best[pv][0] < 0, "prose says Priebe grows slower than his own baseline — re-verify"

# --------------------------------------------------------------------------------- sanity
print(f"[sanity] hot: {label_of[center]} {hot_year} S={hot_S:.1f} W {w_prev}->{w_hot} "
      f"({len(members)} members, {papers_hot} papers in {hot_year}), "
      f"W_{YEARS[-1]}={series[-1][1]}, gap_last={gap_last:.1f}")
print("[sanity] members:", ", ".join(label_of[m] for m in members))
print("[sanity] runners:", "; ".join(
    f"{label_of[r['id']]} {r['year']} S={r['S']} ({r['w_prev']}->{r['w']})" for r in runners))
print(f"[sanity] Priebe: rank {p_rank}/{len(ranked)}, best S={best[pv][0]:.2f} "
      f"in {YEARS[best[pv][1]]}, W last 3 = {[round(float(W[t, pv]), 2) for t in range(T - 3, T)]}")

# ------------------------------------------------------------------------------- envelope
assert label_of[center] == "Can Rager", "winner changed — re-verify and rewrite the headline"
headline = (
    f"The map is hottest around <strong>{label_of[center]}</strong>, whose corner's "
    f"collaboration weight jumped from {w_prev:.1f} to <strong>{w_hot:.1f} in {hot_year}</strong> "
    f"— <strong>{papers_hot} distinct papers</strong> among the group in a single year — "
    f"and it hasn't cooled since."
)

payload = {
    "slug": SLUG,
    "title": "Where it's heating up",
    "headline": headline,
    "data": {
        "years": YEARS,
        "hot": {
            "center": center,
            "year": hot_year,
            "S": round(hot_S, 1),
            "D": d_hot,
            "w_prev": w_prev,
            "w": w_hot,
            "papers_hot": papers_hot,
            "members": members,
            "series": series,
            "trend": {"slope": round(float(slope), 4), "intercept": round(float(intercept), 4)},
            "gap_last": round(gap_last, 1),
        },
        "runners": runners,
        "priebe": {"id": V[pv], "rank": p_rank, "n_scored": len(ranked)},
    },
}

blob = json.dumps(payload, separators=(",", ":"))
assert len(blob) < 300_000
OUT.write_text(blob)
print(f"[{SLUG}] OK {len(blob)/1024:.0f}KB — " + re.sub(r"<[^>]+>", "", headline))
