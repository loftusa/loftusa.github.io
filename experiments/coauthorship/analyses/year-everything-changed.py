# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "networkx", "scikit-learn", "graspologic==3.4.4", "setuptools<81"]
# ///
"""year-everything-changed — iso-mirror change-point on yearly cumulative co-authorship
graphs. Run: cd experiments/coauthorship/analyses && uv run year-everything-changed.py"""
import json
import re
from pathlib import Path

import numpy as np
from graspologic.embed import OmnibusEmbed
from graspologic.utils import pass_to_ranks

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
GRAPH = json.loads((REPO / "assets/data/coauthorship.json").read_text())
DERIVED = HERE / "_derived"
OUT = REPO / "assets/data/analyses" / "year-everything-changed.json"

SLUG = "year-everything-changed"

# ---------------------------------------------------------------- inputs
yearly = json.loads((DERIVED / "yearly.json").read_text())
years: list[int] = [int(y) for y in yearly["years"]]
T = len(years)
assert T >= 10 and years == sorted(years), years

# Drop vertices that have zero ties even in the FINAL cumulative graph (13 list
# members with no co-authorship edges here). They would sit motionless at the
# origin in every year and dilute the mean displacement. Keep everyone else,
# remapping edge indices into the kept order.
raw_order: list[str] = yearly["vertex_order"]
ever = {v for i, j, _ in yearly["cumulative"][str(years[-1])] for v in (i, j)}
keep = [k for k in range(len(raw_order)) if k in ever]
remap = {old: new for new, old in enumerate(keep)}
vorder = [raw_order[k] for k in keep]
n = len(vorder)
assert n > 50, n

graph_ids = {nd["id"] for nd in GRAPH["nodes"]}
assert set(vorder) <= graph_ids, sorted(set(vorder) - graph_ids)[:5]
label_of = {nd["id"]: nd["label"] for nd in GRAPH["nodes"]}
comm_of = {nd["id"]: nd.get("community", -1) for nd in GRAPH["nodes"]}


def reindexed(edges: list[list]) -> list[list]:
    out = [[remap[i], remap[j], w] for i, j, w in edges]
    assert all(0 <= i < n and 0 <= j < n and i != j and w > 0 for i, j, w in out)
    return out


def adj(edges: list[list]) -> np.ndarray:
    A = np.zeros((n, n))
    for i, j, w in reindexed(edges):
        A[i, j] = w
        A[j, i] = w
    return A


cum = {y: adj(yearly["cumulative"][str(y)]) for y in years}
# every kept vertex has at least one tie by the final year
assert ((cum[years[-1]] > 0).sum(axis=1) > 0).all()


def mirror_curve(stack: list[np.ndarray]) -> np.ndarray:
    """OMNI all years jointly into one aligned 3-d space; mean per-vertex
    year-over-year displacement. algorithm='full' = plain LAPACK SVD, deterministic."""
    X = np.asarray(
        OmnibusEmbed(n_components=3, algorithm="full", check_lcc=False).fit_transform(stack)
    )
    assert X.shape == (T, n, 3), X.shape
    d = np.linalg.norm(np.diff(X, axis=0), axis=2)  # (T-1, n)
    return d, d.mean(axis=1)


myears = years[1:]

# We ship the PURE-TOPOLOGY change-point: edges as 0/1, so the question is "when did
# the wiring diagram reorganize most?", not "when did publication volume spike?". The
# binarized mirror is also the curve we draw, so the headline ratio matches the plot.
disp, mirror = mirror_curve([(cum[y] > 0).astype(float) for y in years])
assert (mirror > 0).all()

# Change-point = largest positive jump in the curve; runner-up = second largest.
jumps = np.diff(mirror)  # jumps[k] = mirror[k+1] - mirror[k] at year myears[k+1]
order = np.argsort(jumps)[::-1]
cp_i, ru_i = int(order[0]) + 1, int(order[1]) + 1
assert jumps[order[0]] > 0 and jumps[order[1]] > 0
cp, runnerup = myears[cp_i], myears[ru_i]
jump_ratio = float(mirror[cp_i] / mirror[cp_i - 1])
assert jump_ratio > 1.5, jump_ratio

# Second lens — raw collaboration VOLUME (rank-transformed fractional weights). This
# reads the same history differently: instead of "when did the wiring reorganize?" it
# asks "when did sheer co-authorship volume spike?". The two lenses now disagree, and
# that disagreement IS the story we disclose: pure structure peaks in 2024 (the Bau
# super-core wiring itself up), raw volume peaks in 2019 (the Vogelstein-lab
# publication burst). We surface the volume change-point in prose, not on the plot.
_, mirror_vol = mirror_curve([pass_to_ranks(cum[y]) for y in years])
cp_vol = myears[int(np.argmax(np.diff(mirror_vol))) + 1]

# Pin both lenses to the years the JS annotations are written against. If a data
# regeneration moves either, fail loudly so the prose can't silently drift. Re-pinned
# 2026-06 after two new core members (Antonio Mari, Eric Todd) split the two methods:
# topology -> 2024, volume -> 2019.
assert cp == 2024, cp
assert runnerup == 2019, runnerup
assert cp_vol == 2019, cp_vol  # the disclosed volume lens

# Top movers at the change-point year. The databases keep two profiles for some
# people ("Samuel Marks" / "Samuel D. Marks") — dedupe by middle-initial-stripped
# name so nobody appears twice in their own panel.
def name_key(node_id: str) -> str:
    return " ".join(t for t in label_of[node_id].lower().replace(".", "").split() if len(t) > 1)


top_movers: list[list] = []
seen_keys: set[str] = set()
for i in np.argsort(disp[cp_i])[::-1]:
    if name_key(vorder[i]) in seen_keys:
        continue
    seen_keys.add(name_key(vorder[i]))
    top_movers.append([vorder[i], round(float(disp[cp_i][i]), 4)])
    if len(top_movers) == 4:
        break
assert len(top_movers) == 4 and all(tid in graph_ids for tid, _ in top_movers)

# ------------------------------------------ sanity prints (dev evidence)
print(f"[{SLUG}] mirror curve:")
for k, y in enumerate(myears):
    bar = "#" * int(round(mirror[k] / mirror.max() * 40))
    mark = " <-- change-point" if y == cp else (" <-- runner-up" if y == runnerup else "")
    print(f"  {y}  m={mirror[k]:.5f}  {bar}{mark}")
print(f"[{SLUG}] cp={cp} (jump {jumps[cp_i-1]:+.5f}, ratio {jump_ratio:.2f}x), runner-up={runnerup}; volume lens says {cp_vol}")
print(f"[{SLUG}] top movers at {cp}: " + ", ".join(f"{label_of[i]} ({v:.3f})" for i, v in top_movers))

# Which lab pairs do the NEW ties of a year connect? (annotation evidence)
comm_label = {c["id"]: c["label"] for c in GRAPH["communities"]}


def new_ties_by_pair(year: int) -> dict[tuple, int]:
    """Lab-pair counts of ties genuinely NEW that year (absent from the prior cumulative graph)."""
    prior = {tuple(sorted((i, j))) for i, j, _ in reindexed(yearly["cumulative"][str(year - 1)])} \
        if str(year - 1) in yearly["cumulative"] else set()
    counts: dict[tuple, int] = {}
    for i, j, _ in reindexed(yearly["per_year"][str(year)]):
        if tuple(sorted((i, j))) in prior:
            continue
        a, b = sorted([comm_of.get(vorder[i], -1), comm_of.get(vorder[j], -1)])
        counts[(a, b)] = counts.get((a, b), 0) + 1
    return counts


cp_pairs = new_ties_by_pair(cp)
print(f"[{SLUG}] new ties in {cp} by lab pair:")
for (a, b), c in sorted(cp_pairs.items(), key=lambda kv: -kv[1]):
    print(f"  {comm_label.get(a, 'periphery')} - {comm_label.get(b, 'periphery')}: {c}")

# Data-verified annotation sublines (asserted, so the prose can't drift from the data).
# These are integer pair counts (tie present/absent), so the binarized change of lens
# doesn't touch them. In 2024 the Bau lab wires up internally and throws bridges to the
# other two labs: the subline reports the Bau-internal burst and the total bridging ties
# out of the Bau lab.
assert max(cp_pairs, key=cp_pairs.get) == (1, 1), cp_pairs  # Bau-internal burst dominates
cross = cp_pairs.get((0, 1), 0) + cp_pairs.get((1, 2), 0)
assert cross >= 10, cp_pairs  # bridges out are real
anno_sub = (
    f"{cp_pairs[(1, 1)]} new ties inside the Bau lab; "
    f"{cross} more bridging to the other labs"
)

ru_pairs = new_ties_by_pair(runnerup)
assert set(ru_pairs) == {(2, 2)}, ru_pairs  # runner-up year is purely Vogelstein-internal
runnerup_sub = "the first core: the Vogelstein lab"

# ------------------------------------------------------- small multiples
mini_years = list(years[1::2])  # every other year: 2016, 2018, ... 2026
if cp not in mini_years:
    nearest = min(mini_years, key=lambda y: abs(y - cp))
    mini_years[mini_years.index(nearest)] = cp
    mini_years = sorted(set(mini_years))
assert cp in mini_years and 5 <= len(mini_years) <= 7

minis = []
for y in mini_years:
    edges = [[int(i), int(j)] for i, j, _ in reindexed(yearly["cumulative"][str(y)])]
    active = sorted({v for e in edges for v in e})
    assert all(0 <= v < n for v in active)
    minis.append({"year": y, "edges": edges, "active": active})

edge_counts = [len(yearly["cumulative"][str(y)]) for y in myears]
assert edge_counts == sorted(edge_counts), "cumulative counts must be monotone"

# ---------------------------------------------------------------- payload
ratio_str = f"{jump_ratio:.1f}"
payload = {
    "slug": SLUG,
    "title": "The year everything changed",
    "headline": (
        f"In <strong>{cp}</strong> the network&rsquo;s wiring moved "
        f"<strong>{ratio_str}&times;</strong> farther from its own past than the year "
        "before &mdash; the year the interpretability super-core formed. (Read by raw "
        f"collaboration volume instead, the biggest jump is {cp_vol}, the Vogelstein-lab "
        "publication burst.)"
    ),
    "data": {
        "vertex_order": vorder,
        "years": myears,
        "mirror": [round(float(m), 5) for m in mirror],
        "edge_counts": edge_counts,
        "changepoint": cp,
        "runnerup": runnerup,
        "jump_ratio": round(jump_ratio, 2),
        "anno_sub": anno_sub,
        "runnerup_sub": runnerup_sub,
        "top_movers": top_movers,
        "minis": minis,
    },
}

blob = json.dumps(payload, separators=(",", ":"))
assert len(blob) < 300_000, len(blob)
OUT.write_text(blob)
print(f"[{SLUG}] OK {len(blob)/1024:.0f}KB — " + re.sub(r"<[^>]+>|&\w+;", "", payload["headline"]))
