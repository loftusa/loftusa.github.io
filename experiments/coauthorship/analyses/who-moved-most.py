# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "networkx", "scikit-learn", "graspologic==3.4.4", "setuptools<81"]
# ///
"""who-moved-most — OMNI temporal drift. Run: cd experiments/coauthorship/analyses && uv run who-moved-most.py"""
import json
import re
from pathlib import Path

import numpy as np
from graspologic.embed import OmnibusEmbed

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
GRAPH = json.loads((REPO / "assets/data/coauthorship.json").read_text())
DERIVED = HERE / "_derived"
OUT = REPO / "assets/data/analyses" / "who-moved-most.json"

SLUG = "who-moved-most"
N_COMPONENTS = 4
FIRST_YEAR = 2019  # cumulative graphs before 2019 have <= 13 edges: too sparse to embed
TOP_K = 12

# ---- cumulative weighted adjacencies on the fixed vertex set ----
yearly = json.loads((DERIVED / "yearly.json").read_text())
vorder: list[str] = yearly["vertex_order"]
n = len(vorder)
assert n > 0 and len(set(vorder)) == n
years = [y for y in yearly["years"] if y >= FIRST_YEAR]
assert years == sorted(years) and len(years) >= 3, years

node_by_id = {nd["id"]: nd for nd in GRAPH["nodes"]}
assert all(v in node_by_id for v in vorder), "vertex_order id missing from shipped graph"

mats: list[np.ndarray] = []
nbrs_by_year: dict[int, list[set[int]]] = {}  # year -> [neighbor idx set per vertex]
for yr in years:
    A = np.zeros((n, n))
    nbrs: list[set[int]] = [set() for _ in range(n)]
    for i, j, w in yearly["cumulative"][str(yr)]:
        assert 0 <= i < n and 0 <= j < n and i != j and w > 0, (i, j, w)
        A[i, j] = A[j, i] = w
        nbrs[i].add(j)
        nbrs[j].add(i)
    assert np.allclose(A, A.T)
    mats.append(A)
    nbrs_by_year[yr] = nbrs
for a, b in zip(mats, mats[1:]):
    assert np.all(b >= a - 1e-12), "cumulative adjacencies must be monotone"

# ---- omnibus embedding: every year jointly, ONE shared latent space ----
# algorithm="full" = deterministic dense SVD, so re-runs are byte-identical
omni = OmnibusEmbed(n_components=N_COMPONENTS, algorithm="full", check_lcc=False, svd_seed=0)
X = np.asarray(omni.fit_transform(mats))  # (n_years, n, d)
assert X.shape == (len(years), n, N_COMPONENTS), X.shape
assert np.all(np.isfinite(X))

# ---- per-node drift between consecutive years ----
drift = np.linalg.norm(np.diff(X, axis=0), axis=2)  # (n_years-1, n)
assert drift.shape == (len(years) - 1, n) and np.all(drift >= 0)
drift_years = years[1:]
total = drift.sum(axis=0)

# ---- rank: roster members with at least one co-authorship edge ----
eligible = [i for i, v in enumerate(vorder)
            if node_by_id[v]["is_list"] and node_by_id[v]["degree"] > 0]
assert len(eligible) >= 20, len(eligible)
order = sorted(eligible, key=lambda i: -total[i])
top = order[:TOP_K]
median_total = float(np.median(total[eligible]))

def wdeg(yr: int, i: int) -> float:
    """Weighted degree (sum of fractional co-authorship tie strengths) in the cumulative graph at yr."""
    return float(mats[years.index(yr)][i].sum())

movers = []
for i in top:
    per_year = []
    for t, yr in enumerate(drift_years):
        prev = nbrs_by_year[years[years.index(yr) - 1]][i]
        new_ties = len(nbrs_by_year[yr][i] - prev)  # node-level: new edges on this map
        per_year.append({"year": yr, "d": round(float(drift[t, i]), 4),
                         "new": new_ties, "w": round(wdeg(yr, i), 1)})
    # cumulative neighbors only accrete, so new-tie counts must telescope
    assert sum(p["new"] for p in per_year) == \
        len(nbrs_by_year[years[-1]][i]) - len(nbrs_by_year[years[0]][i])
    b = int(np.argmax(drift[:, i]))
    movers.append({
        "id": vorder[i],
        "total": round(float(total[i]), 4),
        "breakout": drift_years[b],
        "w0": round(wdeg(years[0], i), 1),
        "per_year": per_year,
    })
assert all(m["id"] in node_by_id for m in movers)
assert movers == sorted(movers, key=lambda m: -m["total"])
assert all(node_by_id[m["id"]]["is_list"] for m in movers)

# ---- headline + top-row annotation, straight from the data ----
top1 = movers[0]
top1_label = node_by_id[top1["id"]]["label"]
b_idx = next(t for t, p in enumerate(top1["per_year"]) if p["year"] == top1["breakout"])
b_entry = top1["per_year"][b_idx]
w_before = top1["w0"] if b_idx == 0 else top1["per_year"][b_idx - 1]["w"]
assert median_total > 0, median_total
ratio = top1["total"] / median_total
if b_entry["new"] > 0:
    annotation = (f"{top1['breakout']} — +{b_entry['new']} co-author ties, "
                  f"total tie strength {w_before:g}→{b_entry['w']:g}")
    headline = (f"<strong>{top1_label}</strong> moved the most — one <strong>{top1['breakout']}</strong> "
                f"breakout year added <strong>{b_entry['new']} new co-author ties</strong> on this map.")
else:
    annotation = (f"{top1['breakout']} — no new ties, existing ones deepened: "
                  f"total tie strength {w_before:g}→{b_entry['w']:g}")
    headline = (f"<strong>{top1_label}</strong> moved the most — <strong>{ratio:.0f}×</strong> the median "
                f"roster drift, biggest jump in <strong>{top1['breakout']}</strong> from deepening "
                f"existing ties, not adding new ones.")

payload = {
    "slug": SLUG,
    "title": "Who moved the most",
    "headline": headline,
    "data": {
        "years": years,
        "drift_years": drift_years,
        "n_eligible": len(eligible),
        "median_total": round(median_total, 4),
        "annotation": annotation,
        "movers": movers,
    },
}

blob = json.dumps(payload, separators=(",", ":"))
assert len(blob) < 300_000, len(blob)
OUT.write_text(blob)

print("eligible:", len(eligible), "| median total drift:", round(median_total, 4))
for m in movers:
    print(f"  {node_by_id[m['id']]['label']:<28} total={m['total']:.3f}  breakout={m['breakout']}")
print(f"[{SLUG}] OK {len(blob)/1024:.0f}KB — " + re.sub(r"<[^>]+>", "", headline))
