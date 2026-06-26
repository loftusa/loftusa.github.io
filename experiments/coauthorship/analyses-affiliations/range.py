# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""range — perplexity of each person's affiliation-type distribution (Hill N1).
Run: cd experiments/coauthorship/analyses-affiliations && uv run range.py"""
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
assert (REPO / "assets").exists(), f"REPO mis-resolved: {REPO}"

AFF = json.loads((REPO / "assets/data/affiliations.json").read_text())
GRAPH = json.loads((REPO / "assets/data/coauthorship.json").read_text())
OUT = REPO / "assets/data/analyses-affiliations" / "range.json"

SLUG = "range"
TYPES = {"lab", "program", "company", "community", "university"}
RING_R = 8.0  # px radius of the tie fan-out ring

# ---- inputs: shipped canonical data only -----------------------------------------------
graph_ids = {n["id"] for n in GRAPH["nodes"]}
org_type = {o["id"]: o["type"] for o in AFF["orgs"]}
assert set(org_type.values()) == TYPES, f"unexpected org types: {set(org_type.values())}"

people_ids = {p["id"] for p in AFF["people"]}
assert len(people_ids) == 52
assert people_ids <= graph_ids, f"AFF people missing from graph: {people_ids - graph_ids}"

person_orgs: dict[str, set[str]] = defaultdict(set)
for link in AFF["links"]:
    assert link["person"] in people_ids, f"link person not in people: {link['person']}"
    assert link["org"] in org_type, f"link org not in orgs: {link['org']}"
    assert link["org"] not in person_orgs[link["person"]], (
        f"shipped links not person-org deduped: {link['person']} / {link['org']}"
    )
    person_orgs[link["person"]].add(link["org"])
for pid in people_ids - set(person_orgs):
    person_orgs[pid] = set()                 # roster member with no chapters yet — honest zero

# ---- per person: type counts, Shannon H, Hill N1 = exp(H) ------------------------------
stats: dict[str, dict] = {}
for pid, orgs in person_orgs.items():
    counts = Counter(org_type[o] for o in orgs)
    n = sum(counts.values())
    h = -sum((c / n) * math.log(c / n) for c in counts.values()) if n else 0.0
    n1 = math.exp(h)
    assert 1.0 <= n1 <= 5.0 + 1e-12, f"{pid}: N1={n1} out of [1,5]"
    stats[pid] = {"n": n, "n1": n1, "counts": counts}

# ---- headline / spec assertions --------------------------------------------------------
ge3 = sorted(p for p, s in stats.items() if len(s["counts"]) >= 3)
assert ge3, 'nobody with >=3 types?'

top_id = max(stats, key=lambda p: stats[p]["n1"])


tie_379 = sorted(p for p, s in stats.items() if round(s["n1"], 2) == 3.79)

zeno = stats["zeno kujawa"]

# ---- deterministic fan-out for exact (n, N1) ties: alphabetical index on an 8-ring -----
tie_groups: dict[tuple, list[str]] = defaultdict(list)
for pid, s in stats.items():
    tie_groups[(s["n"], round(s["n1"], 9))].append(pid)
offset = {}
for key, members in tie_groups.items():
    members.sort()
    assert len(members) <= 12, f"tie group {key} too large to fan out legibly: {members}"
    for k, pid in enumerate(members):
        if len(members) == 1:
            offset[pid] = (0.0, 0.0)
        else:
            a = 2 * math.pi * k / max(len(members), 8)   # even spacing, 8-point look up to 8
            offset[pid] = (round(RING_R * math.cos(a), 2), round(RING_R * math.sin(a), 2))

# ---- payload ---------------------------------------------------------------------------
people = []
for pid in sorted(stats):
    s = stats[pid]
    dx, dy = offset[pid]
    people.append({
        "id": pid,
        "n": s["n"],
        "n1": round(s["n1"], 4),
        "types": {t: s["counts"][t] for t in sorted(s["counts"])},
        "low": s["n"] < 3,
        "dx": dx,
        "dy": dy,
    })

_top_label = next(p["label"] for p in AFF["people"] if p["id"] == top_id)
headline = (
    f"<strong>{len(ge3)}</strong> of the {len(people_ids)} people here have lived in at least "
    f"three different kinds of institution — and {_top_label}’s record runs at perplexity "
    f"<strong>{stats[top_id]['n1']:.1f}</strong> out of a possible 5."
)
payload = {
    "slug": SLUG,
    "title": "Range",
    "headline": headline,
    "data": {
        "people": people,
        "n_ge3_types": len(ge3),
        "max_n1": {"id": top_id, "n1": round(stats[top_id]["n1"], 4)},
        "annotated": ["alex loftus", "roy rinberg", "zeno kujawa", "jeremy howard"],
    },
}

blob = json.dumps(payload, separators=(",", ":"), ensure_ascii=False)
assert len(blob) < 300_000, f"payload too big: {len(blob)}"
OUT.write_text(blob)

import re as _re
print(f"[{SLUG}] OK {len(blob)/1024:.0f}KB — " + _re.sub(r"<[^>]+>", "", headline))
