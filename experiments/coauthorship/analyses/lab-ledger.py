# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "networkx", "scikit-learn", "graspologic==3.4.4", "setuptools<81"]
# ///
"""lab-ledger — DCSBM block matrix: lab-to-lab trade after degree correction.
Run: cd experiments/coauthorship/analyses && uv run lab-ledger.py"""
import json
import re
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
GRAPH = json.loads((REPO / "public/assets/data/coauthorship.json").read_text())
DERIVED = HERE / "_derived"  # unused here; kept per contract template
OUT = REPO / "public/assets/data/analyses" / "lab-ledger.json"

SLUG = "lab-ledger"
LABS = [0, 1, 2]
SHORT = {0: "EleutherAI", 1: "Bau lab", 2: "Vogelstein lab"}

assert GRAPH["meta"]["weighting"] == "fractional", "expected fractional edge weights"

community = {n["id"]: n["community"] for n in GRAPH["nodes"]}
label = {n["id"]: n["label"] for n in GRAPH["nodes"]}
assert len(community) == len(GRAPH["nodes"]), "duplicate node ids"
lab_label = {c["id"]: c["label"] for c in GRAPH["communities"]}
assert set(lab_label) == set(LABS), f"unexpected community ids {lab_label}"

n_members = {a: sum(1 for c in community.values() if c == a) for a in LABS}
# sizes include the Feucht editorial override (overrides.json: she is a Bau-lab student)
assert n_members == {0: 22, 1: 56, 2: 29}, f"community sizes shifted: {n_members}"

# ---- observed block matrix W (exclude community -1; diagonal counted once) ----------
# W sums fractional edge weights (each paper adds 1/n_authors per pair).
# edges_n counts integer direct edges per cell; cell_papers tracks distinct titles.
W = [[0.0] * 3 for _ in range(3)]
edges_n = [[0] * 3 for _ in range(3)]
cell_papers = defaultdict(set)  # (a,b) a<=b -> distinct paper titles
carrier_w = defaultdict(lambda: defaultdict(float))  # (a,b) a<b -> person -> weight
n_excluded = 0
for l in GRAPH["links"]:
    cu, cv = community[l["source"]], community[l["target"]]
    assert l["weight"] > 0
    assert l["n_papers"] >= 1
    if cu < 0 or cv < 0:
        n_excluded += 1
        continue
    a, b = sorted((cu, cv))
    W[a][b] += l["weight"]
    edges_n[a][b] += 1
    if a != b:
        W[b][a] += l["weight"]
        edges_n[b][a] += 1
        carrier_w[(a, b)][l["source"]] += l["weight"]
        carrier_w[(a, b)][l["target"]] += l["weight"]
    cell_papers[(a, b)].update(l["papers"])
assert n_excluded == 0, "periphery (-1) nodes hold edges now — update prose caveats"
assert all(W[a][b] == W[b][a] for a in LABS for b in LABS), "W not symmetric"
n_cell_edges = sum(edges_n[a][b] for a in LABS for b in LABS if a <= b)
assert n_cell_edges == len(GRAPH["links"]), "edge count mismatch vs shipped graph"

# ---- degree-corrected expectation: E_ab = d_a d_b / 2m off-diag, d_a^2 / 4m on ------
deg = [2 * W[a][a] + sum(W[a][b] for b in LABS if b != a) for a in LABS]
m = sum(W[a][b] for a in LABS for b in LABS if a <= b)
assert abs(sum(deg) - 2 * m) < 1e-9, "handshake check failed"
E = [
    [deg[a] * deg[b] / (2 * m) if a != b else deg[a] ** 2 / (4 * m) for b in LABS]
    for a in LABS
]
R = [[W[a][b] / E[a][b] for b in LABS] for a in LABS]
percap = [
    [
        W[a][b] / (n_members[a] * n_members[b])
        if a != b
        else W[a][a] / (n_members[a] * (n_members[a] - 1) / 2)
        for b in LABS
    ]
    for a in LABS
]
papers_n = [[len(cell_papers[tuple(sorted((a, b)))]) for b in LABS] for a in LABS]
print(f"[{SLUG}] W={[[round(x, 4) for x in row] for row in W]}")
print(f"[{SLUG}] deg={[round(x, 4) for x in deg]} m={round(m, 4)}")
print(f"[{SLUG}] ratio={[[round(x, 3) for x in row] for row in R]}")
print(f"[{SLUG}] percap={[[round(x, 5) for x in row] for row in percap]}")
print(f"[{SLUG}] direct edges per cell={edges_n}")
print(f"[{SLUG}] distinct papers per cell={papers_n}")

# ---- top carriers of each nonzero off-diagonal cell ---------------------------------
carriers = {}
for (a, b), per_person in sorted(carrier_w.items()):
    total = W[a][b]
    top = sorted(per_person.items(), key=lambda kv: (-kv[1], kv[0]))[:3]
    carriers[f"{a}{b}"] = [
        {"id": pid, "w": round(w, 3), "share": round(w / total, 4)} for pid, w in top
    ]
    for pid, _ in top:
        assert pid in community, f"carrier {pid} not in GRAPH nodes"

# ---- verify the headline facts -------------------------------------------------------
assert W[0][2] == 0.0, "EleutherAI–Vogelstein cell no longer zero — rewrite headline"
assert edges_n[0][2] == 0, "direct EleutherAI–Vogelstein edges appeared — rewrite prose"
fold = R[0][1] / R[1][2]
assert 15.0 <= fold < 40.0, f"seam fold left its regime — rewrite headline: {fold}"
top01 = carriers["01"][0]
assert top01["id"] == "kola ayonrinde", f"unexpected top carrier {top01}"
assert 0.20 <= top01["share"] <= 0.55, f"top-carrier share left its regime: {top01}"
c12 = carriers["12"][0]
assert c12["id"] == "alex loftus" and c12["share"] == 1.0, c12
# evidence-line numbers — the JS intro prose bakes these; keep it in sync with this print
print(f"[{SLUG}] seam evidence: edges={edges_n[0][1]} weight={W[0][1]:.2f} papers={papers_n[0][1]}")

headline = (
    "Strip out what hub sizes alone would produce and one open channel dominates the "
    f"ledger: EleutherAI and the Bau lab trade <strong>{round(fold)}&times;</strong> more "
    f"intensely than the Bau and Vogelstein labs do &mdash; and "
    f"<strong>{label[top01['id']]}</strong> alone accounts for "
    f"{round(100 * top01['share'])}% of it."
)

payload = {
    "slug": SLUG,
    "title": "The lab-to-lab ledger",
    "headline": headline,
    "data": {
        "labs": [
            {
                "id": a,
                "label": lab_label[a],
                "short": SHORT[a],
                "n": n_members[a],
                "degree": round(deg[a], 2),
            }
            for a in LABS
        ],
        "m": round(m, 2),
        "observed": [[round(x, 2) for x in row] for row in W],
        "edges": edges_n,
        "expected": [[round(x, 1) for x in row] for row in E],
        "ratio": [[round(x, 3) for x in row] for row in R],
        "percap": [[round(x, 5) for x in row] for row in percap],
        "papers": papers_n,
        "carriers": carriers,
        "fold": round(fold, 1),
    },
}

blob = json.dumps(payload, separators=(",", ":"))
assert len(blob) < 300_000, f"payload too big: {len(blob)}"
OUT.write_text(blob)
print(f"[{SLUG}] OK {len(blob)/1024:.0f}KB — " + re.sub(r"<[^>]+>", "", headline))
