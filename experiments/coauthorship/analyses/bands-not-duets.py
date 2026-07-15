# /// script
# requires-python = ">=3.10"
# dependencies = ["setuptools<81"]
# ///
"""bands-not-duets — recurring author-set mining (standing teams in the hypergraph).
Run: cd experiments/coauthorship/analyses && uv run bands-not-duets.py"""
import json
import re
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
GRAPH = json.loads((REPO / "public/assets/data/coauthorship.json").read_text())
DERIVED = HERE / "_derived"
OUT = REPO / "public/assets/data/analyses" / "bands-not-duets.json"

SLUG = "bands-not-duets"
N_SHIP = 10  # ship top ~10 bands

nodes = {n["id"]: n for n in GRAPH["nodes"]}
label_of = {nid: n["label"] for nid, n in nodes.items()}

# ---- collapse split database identities of one human onto a canonical node -------------
# The two source databases occasionally keep two profiles for the same person
# (e.g. "J. Vogelstein" vs "Joshua T Vogelstein"). Detected by (last name, first
# initial); canonical = the higher-degree spelling. Verified: every pair below is
# co-listed on many shared papers, i.e. genuinely one person.
alias_groups = defaultdict(list)
for nid in nodes:
    toks = nid.split()
    alias_groups[(toks[-1], toks[0][0])].append(nid)
CANON = {}
for key, ids in alias_groups.items():
    if len(ids) < 2:
        continue
    best = max(ids, key=lambda i: (nodes[i]["degree"], len(nodes[i]["label"]), i))
    for i in ids:
        CANON[i] = best
canon = lambda nid: CANON.get(nid, nid)
print(f"[{SLUG}] alias collapses: " + ", ".join(
    f"{i}->{CANON[i]}" for i in CANON if CANON[i] != i))

# ---- papers -> deduped "works" (member-set, year) --------------------------------------
papers = json.loads((DERIVED / "papers.json").read_text())
# eLife "Author response:" / "Decision letter:" entries duplicate the real paper under a
# prefixed title; strip the prefix before deduping so they fold into the work they review.
ELIFE = re.compile(
    r"^\s*(author response|decision letter|reviewer response|editor.?s evaluation)\s*[:\-]\s*",
    re.I)


def tkey(title: str) -> str:
    title = ELIFE.sub("", title)
    return " ".join("".join(c for c in title.lower() if c.isalnum() or c == " ").split())


by_title = defaultdict(list)
for p in papers:
    if p["big"]:  # big = >25 authors; shipped graph draws no list<->list edge from these
        continue
    by_title[tkey(p["title"])].append(p)

works = []
for versions in by_title.values():
    members = frozenset(canon(m) for v in versions for m in v["members"])
    yrs = [v["year"] for v in versions if v["year"]]
    if not yrs:
        continue
    works.append({
        "title": min((v["title"] for v in versions), key=len),  # shortest = cleanest
        "year": min(yrs),
        "members": members,
    })

# Papers with 3..10 graph members (a SET of people, not a pair) are the hyperedges we mine.
cand = [w for w in works if 3 <= len(w["members"]) <= 10]
print(f"[{SLUG}] works after title-dedup: {len(works)}; "
      f"hyperedges with 3..10 members: {len(cand)}")
assert all(w["year"] for w in cand), "candidate work missing a year"

# ---- mine bands: an exact member-set recurring on >=2 distinct works --------------------
exact = defaultdict(list)
for w in cand:
    exact[w["members"]].append(w)
seed_sets = [s for s, ws in exact.items() if len(ws) >= 2]
print(f"[{SLUG}] exact recurring member-sets (>=2 works): {len(seed_sets)}")

bands = []
for s in seed_sets:
    # absorb near-misses: a work whose set is exactly s, or a superset adding <=1 person,
    # counts toward this band's run (the standing core plus an occasional guest).
    run = [w for w in cand
           if w["members"] == s or (w["members"] > s and len(w["members"] - s) == 1)]
    run.sort(key=lambda w: (w["year"], w["title"]))
    years = [w["year"] for w in run]
    bands.append({
        "set": s,
        "run": run,
        "n": len(run),
        "exact": len(exact[s]),
        "span": [min(years), max(years)],
    })

# Rank by run length, then year span (longest-standing first).
bands.sort(key=lambda b: (-b["n"], -(b["span"][1] - b["span"][0]), sorted(b["set"])))
top = bands[:N_SHIP]
assert len(top) == N_SHIP, f"only {len(top)} bands found"

# ---- assemble shipped data -------------------------------------------------------------
def member_records(s):
    # order by community (groups lab colors) then label
    return sorted(s, key=lambda nid: (nodes[nid]["community"], label_of[nid]))


bands_out = []
all_years = []
for b in top:
    core = set(b["set"])
    members = member_records(b["set"])
    for nid in members:
        assert nid in nodes, f"shipped id {nid} not in GRAPH nodes"
    papers_out = []
    for w in b["run"]:
        guest = w["members"] - core
        assert len(guest) <= 1
        papers_out.append({
            "title": w["title"],
            "year": w["year"],
            "guest": label_of[next(iter(guest))] if guest else None,
        })
        all_years.append(w["year"])
    bands_out.append({
        "members": members,
        "papers": papers_out,
        "n": b["n"],
        "exact": b["exact"],
        "span": b["span"],
    })

year_lo, year_hi = min(all_years), max(all_years)

# headline names: first + last token of the longest band's members, lab-ordered by size
lead = bands_out[0]
short = lambda lab: lab.split()[0] + " " + lab.split()[-1]
lead_names = [short(label_of[m]) for m in lead["members"]]
assert len(lead_names) == 3, "expected the lead band to be a trio"
names_str = ", ".join(lead_names[:-1]) + " and " + lead_names[-1]
lo, hi = lead["span"]

# how many of the shipped bands sit entirely inside one lab (single-community cores)?
single_lab = sum(1 for b in bands_out
                 if len({nodes[m]["community"] for m in b["members"]}) == 1)

headline = (f"Edges only see pairs, but the longest-running team is a trio: "
            f"<strong>{names_str}</strong> published as a band at least "
            f"<strong>{lead['n']} times</strong> across {lo}–{hi} — the standing team a "
            f"pairwise graph can’t see.")

payload = {
    "slug": SLUG,
    "title": "Bands, not duets",
    "headline": headline,
    "data": {
        "bands": bands_out,
        "years": [year_lo, year_hi],
        "single_lab": single_lab,
        "n_bands": len(bands_out),
    },
}

# ---- sanity prints ---------------------------------------------------------------------
print(f"[{SLUG}] year axis: {year_lo}-{year_hi}; "
      f"{single_lab}/{len(bands_out)} bands are single-lab")
for i, b in enumerate(bands_out, 1):
    labs = [label_of[m] for m in b["members"]]
    print(f"  #{i:2d} n={b['n']} exact={b['exact']} span={b['span']} :: {labs}")

blob = json.dumps(payload, separators=(",", ":"))
assert len(blob) < 300_000, len(blob)
OUT.write_text(blob)
plain = re.sub("<[^>]+>", "", headline)
print(f"[{SLUG}] OK {len(blob)/1024:.0f}KB — {plain}")
