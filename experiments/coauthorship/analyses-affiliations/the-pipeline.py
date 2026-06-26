# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""the-pipeline — careers as a five-token language: org-type transition matrix.
Run: cd experiments/coauthorship/analyses-affiliations && uv run the-pipeline.py"""
import json
import re
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
assert (REPO / "assets").exists(), f"REPO mis-anchored: {REPO}"

AFF = json.loads((REPO / "assets/data/affiliations.json").read_text())
GRAPH = json.loads((REPO / "assets/data/coauthorship.json").read_text())
OUT = REPO / "assets/data/analyses-affiliations" / "the-pipeline.json"



def start_year(y):
    m = re.match(r"^\s*(\d{4})", y or "")
    return int(m.group(1)) if m else None


GRAPH_IDS = {n["id"] for n in GRAPH["nodes"]}
ORG_TYPE = {o["id"]: o["type"] for o in AFF["orgs"]}
ORG_LABEL = {o["id"]: o["label"] for o in AFF["orgs"]}

TYPES = ["university", "lab", "company", "program", "community"]
TIDX = {t: i for i, t in enumerate(TYPES)}
assert set(ORG_TYPE.values()) == set(TYPES), sorted(set(ORG_TYPE.values()))

# ---- dated links per person, sorted by (start_year, org_id) ----
links = AFF["links"]
for l in links:
    assert l["org"] in ORG_TYPE, f"org not in AFF orgs: {l['org']}"
    assert l["person"] in GRAPH_IDS, f"person not a graph node: {l['person']}"
dated = [l for l in links if start_year(l.get("years")) is not None]
assert dated and len(dated) <= len(links)

by_person = defaultdict(list)
for l in dated:
    by_person[l["person"]].append(l)
for ls in by_person.values():
    ls.sort(key=lambda l: (start_year(l["years"]), l["org"]))

# ---- transitions: consecutive pairs with STRICTLY increasing start years ----
counts = [[0] * 5 for _ in range(5)]
cells = defaultdict(list)  # "from_type>to_type" -> [(person, from_org, to_org, y0, y1)]
n_simultaneous = 0
people_in_chain = set()
for person in sorted(by_person):
    ls = by_person[person]
    for a, b in zip(ls, ls[1:]):
        y0, y1 = start_year(a["years"]), start_year(b["years"])
        if y1 <= y0:
            assert y1 == y0, "sort violated"
            n_simultaneous += 1
            continue
        ta, tb = ORG_TYPE[a["org"]], ORG_TYPE[b["org"]]
        counts[TIDX[ta]][TIDX[tb]] += 1
        cells[f"{ta}>{tb}"].append((person, a["org"], b["org"], y0, y1))
        people_in_chain.add(person)

n_transitions = sum(map(sum, counts))
row_sums = [sum(r) for r in counts]

# ---- consistency (counts change as people join/edit; only structure is pinned) ----
assert n_transitions == sum(row_sums) > 0
acad_to_co = counts[0][2] + counts[1][2]
co_to_acad = counts[2][0] + counts[2][1]

probs = [
    [round(c / rs, 3) if rs else 0.0 for c in row] for row, rs in zip(counts, row_sums)
]

# ---- per-cell examples: up to 6, sorted (-to_year, person) ----
cells_out = {}
for key, ex in sorted(cells.items()):
    ex.sort(key=lambda t: (-t[4], t[0]))
    cells_out[key] = [
        {"id": p, "from": ORG_LABEL[o0], "to": ORG_LABEL[o1], "y0": y0, "y1": y1}
        for p, o0, o1, y0, y1 in ex[:6]
    ]

# ---- recent crossings: into a company, to_year >= 2024, from university/lab ----
movers = []
for key in ("university>company", "lab>company"):
    for p, o0, o1, y0, y1 in cells.get(key, []):
        if y1 >= 2024:
            movers.append((y1, p, o0, o1))
movers.sort(key=lambda m: (m[0], m[1]))
assert all(y >= 2024 for y, *_ in movers)
movers_out = [
    {"id": p, "year": y, "from": ORG_LABEL[o0], "to": ORG_LABEL[o1]}
    for y, p, o0, o1 in movers
]

# ---- minimap coloring: type of each person's LATEST dated entry ----
latest_type = {p: ORG_TYPE[ls[-1]["org"]] for p, ls in sorted(by_person.items())}
assert set(latest_type) <= GRAPH_IDS
shipped_people = (
    set(latest_type)
    | {e["id"] for ex in cells_out.values() for e in ex}
    | {m["id"] for m in movers_out}
)
assert shipped_people <= GRAPH_IDS, shipped_people - GRAPH_IDS

prog_row, prog_to_co = row_sums[3], counts[3][2]
program_clause = (
    f" — and leaving a program, {'half' if prog_row and prog_to_co * 2 >= prog_row else 'a company is often'}"
    " of all next stops are a company." if prog_row and prog_to_co * 2 >= prog_row else "."
)
headline = (
    f"Across <strong>{n_transitions}</strong> recorded moves people here crossed from academia "
    f"into companies <strong>{acad_to_co}</strong> times and came back {co_to_acad}"
    + program_clause
)

payload = {
    "slug": "the-pipeline",
    "title": "The pipeline",
    "headline": headline,
    "data": {
        "types": TYPES,
        "counts": counts,
        "probs": probs,
        "row_sums": row_sums,
        "n_transitions": n_transitions,
        "n_dated_links": len(dated),
        "n_undated": len(links) - len(dated),
        "n_simultaneous": n_simultaneous,
        "n_people_in_chain": len(people_in_chain),
        "acad_to_co": acad_to_co,
        "co_to_acad": co_to_acad,
        "cells": cells_out,
        "movers": movers_out,
        "latest_type": latest_type,
    },
}

blob = json.dumps(payload, separators=(",", ":"))
assert len(blob) < 300_000, len(blob)
OUT.write_text(blob)
print(
    f"[the-pipeline] OK {len(blob)/1024:.0f}KB — "
    + re.sub(r"<[^>]+>", "", headline)
)
