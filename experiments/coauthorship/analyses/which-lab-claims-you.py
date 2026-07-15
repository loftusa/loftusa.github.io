# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "networkx", "scikit-learn", "graspologic==3.4.4", "setuptools<81"]
# ///
"""which-lab-claims-you — one-hot Graph Encoder Embedding: which lab claims each
off-roster person? Run: cd experiments/coauthorship/analyses && uv run which-lab-claims-you.py"""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
GRAPH = json.loads((REPO / "public/assets/data/coauthorship.json").read_text())
OUT = REPO / "public/assets/data/analyses" / "which-lab-claims-you.json"

SLUG = "which-lab-claims-you"
LABS = [c["label"] for c in GRAPH["communities"]]
assert LABS == ["EleutherAI", "David Bau", "Joshua Vogelstein"], LABS

nodes = GRAPH["nodes"]
ids = [nd["id"] for nd in nodes]
idx = {i: j for j, i in enumerate(ids)}
by_id = {nd["id"]: nd for nd in nodes}
n = len(ids)
assert n == len(set(ids)) == 130, n

# ---- adjacency: every edge drawn on the map (core links + bridge-path links) -------------
# weight = fractional co-authorship strength (each distinct paper adds 1/n_authors per pair);
# the one pair present in both edge lists keeps the max. Alongside it, an integer paper-count
# matrix P (links carry n_papers; path links list their shared papers) for human-readable
# "paper-links" — stats run on A, displayed counts come from P.
assert GRAPH["meta"]["weighting"] == "fractional", GRAPH["meta"].get("weighting")
edges: dict[tuple[str, str], float] = {}
papercnt: dict[tuple[str, str], int] = {}
for e in GRAPH["links"] + GRAPH["path_links"]:
    key = tuple(sorted((e["source"], e["target"])))
    edges[key] = max(edges.get(key, 0.0), float(e["weight"]))
    cnt = int(e["n_papers"]) if "n_papers" in e else len(e["papers"])
    papercnt[key] = max(papercnt.get(key, 0), cnt)
A = np.zeros((n, n))
P = np.zeros((n, n))
for (a, b), w in edges.items():
    assert a in idx and b in idx and w > 0, (a, b, w)
    A[idx[a], idx[b]] = A[idx[b], idx[a]] = w
    P[idx[a], idx[b]] = P[idx[b], idx[a]] = papercnt[(a, b)]
assert np.array_equal(A, A.T) and A.sum() > 0
deg = (A > 0).sum(axis=1)

# ---- labels: the three labs' rosters (list members the page's clustering placed) ---------
seed = {nd["id"]: nd["community"] for nd in nodes if nd["is_list"] and nd["community"] >= 0}
roster_sizes = [sum(1 for k in seed.values() if k == c) for c in range(3)]
assert sum(roster_sizes) == len(seed) and min(roster_sizes) > 0, roster_sizes
# every roster edge that can feed a displayed paper count must actually carry one
seeded_rows = np.array([1.0 if i in seed else 0.0 for i in ids])
assert ((A > 0) & (P == 0) & (seeded_rows[None, :] > 0)).sum() == 0, \
    "roster-touching edge with weight but no paper count"

# ---- GEE: Z = A @ W, W one-hot column-normalized by class size ---------------------------
# One round of label propagation == a 1-layer GNN with frozen one-hot weights.
W = np.zeros((n, 3))
for i, k in seed.items():
    W[idx[i], k] = 1.0
assert np.array_equal(W.sum(axis=0), np.array(roster_sizes, dtype=float))
W /= W.sum(axis=0, keepdims=True)
Z = A @ W
assert Z.shape == (n, 3) and np.isfinite(Z).all()

# ---- query set: everyone not seeded who has at least one drawn edge ----------------------
query = sorted(i for i in ids if i not in seed and deg[idx[i]] > 0)
# every community==-1 node with degree>0 (the bridge connectors) must be queried
connectors = {nd["id"] for nd in nodes if nd["community"] == -1 and deg[idx[nd["id"]]] > 0}
assert connectors <= set(query), connectors - set(query)

people, unplaced = [], []
for i in query:
    z = Z[idx[i]]
    if z.sum() == 0:  # no roster member within one hop: outside the 1-layer receptive field
        unplaced.append(i)
        continue
    aff = z / z.sum()
    raw = [int(P[idx[i]] @ (W[:, k] > 0)) for k in range(3)]  # integer paper-links per roster
    assert sum(raw) > 0 and abs(aff.sum() - 1) < 1e-12
    assert all((r > 0) == (zk > 0) for r, zk in zip(raw, z)), (i, raw, z.tolist())
    people.append({
        "id": i,
        "affinity": [round(float(a), 4) for a in aff],
        "raw": raw,
        "winner": int(aff.argmax()),
        "confidence": round(float(aff.max()), 4),
        "callout": False,
    })

assert all(not by_id[p["id"]]["is_list"] for p in people), "all placed people are off-roster"
assert {p["id"] for p in people} | set(unplaced) == set(query)
counts = [sum(1 for p in people if p["winner"] == k) for k in range(3)]
n_decisive = sum(1 for p in people if p["confidence"] > 0.9999)
both = sum(1 for p in people if sum(r > 0 for r in p["raw"]) >= 2)
all3 = sum(1 for p in people if all(r > 0 for r in p["raw"]))
agree = sum(1 for p in people if by_id[p["id"]]["community"] == p["winner"])
agree_tot = sum(1 for p in people if by_id[p["id"]]["community"] >= 0)

# ---- callouts: strongest claim per lab, the most torn, and two names worth a label -------
def mass(p):  # total paper-links to any roster: evidence behind the verdict
    return sum(p["raw"])

callout_ids = set()
for k in range(3):
    champ = sorted((p for p in people if p["winner"] == k),
                   key=lambda p: (-p["confidence"], -mass(p), p["id"]))[0]
    callout_ids.add(champ["id"])
torn = sorted(people, key=lambda p: (p["confidence"], p["id"]))[:3]
callout_ids |= {p["id"] for p in torn}
callout_ids |= {"david bau", "carey e priebe"} & {p["id"] for p in people}  # the PI + the easter egg
for p in people:
    p["callout"] = p["id"] in callout_ids

# ---- headline facts (asserted, so the sentence can't drift from the data) ----------------
# With Sheridan Feucht editorially assigned to the Bau lab (overrides.json — she is a
# Bau-lab student; the spectral clustering had filed her under EleutherAI), David Bau's
# seed-label affinity goes where it belongs: ~99% the lab that bears his name.
bau = next(p for p in people if p["id"] == "david bau")
assert bau["winner"] == 1 and bau["confidence"] > 0.95, bau
assert bau["affinity"][1] > bau["affinity"][2] >= bau["affinity"][0], bau
bau_own_pct = round(100 * bau["affinity"][1])
priebe = next(p for p in people if p["id"] == "carey e priebe")
assert priebe["winner"] == 2 and priebe["confidence"] == 1.0, priebe
most_torn = torn[0]

headline = (
    f"One round of label propagation hands <strong>{n_decisive}</strong> of "
    f"<strong>{len(people)}</strong> off-roster people to a single lab outright &mdash; and "
    f"the lab that bears <strong>David Bau</strong>&rsquo;s name claims him "
    f"<strong>{bau_own_pct}%</strong>, as it should."
)

payload = {
    "slug": SLUG,
    "title": "Which lab claims you?",
    "headline": headline,
    "data": {
        "labs": LABS,
        "roster_sizes": roster_sizes,
        "counts": counts,
        "n_query": len(query),
        "n_placed": len(people),
        "n_decisive": n_decisive,
        "agreement": {"agree": agree, "total": agree_tot},
        "unplaced": unplaced,
        "people": people,
    },
}

for i in [p["id"] for p in people] + unplaced:
    assert i in by_id, i
blob = json.dumps(payload, separators=(",", ":"))
assert len(blob) < 300_000, len(blob)
OUT.write_text(blob)

print(f"  rosters seeded: {roster_sizes} ({len(seed)} people) | query {len(query)} "
      f"-> placed {len(people)}, unreached {len(unplaced)}")
print(f"  claims by lab: { {LABS[k]: counts[k] for k in range(3)} } | decisive {n_decisive} "
      f"| split across 2 labs: {both} | all 3: {all3}")
print(f"  agreement with page clustering: {agree}/{agree_tot}")
print(f"  most torn: {by_id[most_torn['id']]['label']} {most_torn['affinity']}")
print(f"  david bau: {bau['affinity']} | carey e priebe: {priebe['affinity']}")
import re
print(f"[{SLUG}] OK {len(blob)/1024:.0f}KB — " + re.sub(r"<[^>]+>", "", headline))
