# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""embassies — orgs held by exactly one of the 48. Run: cd experiments/coauthorship/analyses-affiliations && uv run embassies.py"""
import json
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
assert (REPO / "public" / "assets").exists(), f"REPO mis-resolved: {REPO}"

AFF = json.loads((REPO / "public/assets/data/affiliations.json").read_text())
GRAPH = json.loads((REPO / "public/assets/data/coauthorship.json").read_text())
OUT = REPO / "public/assets/data/analyses-affiliations" / "embassies.json"

GRAPH_IDS = {n["id"] for n in GRAPH["nodes"]}
ORG_BY_ID = {o["id"]: o for o in AFF["orgs"]}
PEOPLE = [p["id"] for p in AFF["people"]]
assert len(PEOPLE) >= 48
assert ORG_BY_ID
assert all(p in GRAPH_IDS for p in PEOPLE), "AFF person id missing from graph"

# ---- membership per org (dedupe person-org just in case; verified no dups) ----
members: dict[str, list[str]] = defaultdict(list)
link_of: dict[tuple[str, str], dict] = {}
for l in AFF["links"]:
    assert l["org"] in ORG_BY_ID, f"link to unknown org {l['org']}"
    assert l["person"] in GRAPH_IDS, f"link from unknown person {l['person']}"
    if l["person"] not in members[l["org"]]:
        members[l["org"]].append(l["person"])
    link_of[(l["person"], l["org"])] = l
for org in members:
    members[org].sort()
assert set(members) == set(ORG_BY_ID), "every org must have >=1 member"

# ---- the membership tally: how many people per org ----
tally_counts = Counter(len(ms) for ms in members.values())
n_solo = tally_counts[1]
n_shared = sum(v for k, v in tally_counts.items() if k > 1)
assert n_solo + n_shared == len(ORG_BY_ID)

# ---- solo orgs: type mix, per-person counts ----
solo_orgs = sorted(o for o, ms in members.items() if len(ms) == 1)
solo_types = Counter(ORG_BY_ID[o]["type"] for o in solo_orgs)

holders = Counter(members[o][0] for o in solo_orgs)
assert sorted(p for p, k in holders.items() if k == 5) == [
    "allan deutsch", "jacopo teneggi", "jesse hoogland",
    "liv gorton", "lucia quirke", "roy rinberg",
]
assert len(PEOPLE) - len(holders) == 5  # the five with zero stay a COUNT — never named

# ---- ambassadors: everyone holding >=5 embassies ----
solo_by_person: dict[str, list[str]] = defaultdict(list)
for o in solo_orgs:
    solo_by_person[members[o][0]].append(o)
ambassadors = [
    {"id": p, "k": k, "orgs": sorted(ORG_BY_ID[o]["label"] for o in solo_by_person[p])}
    for p, k in sorted(holders.items(), key=lambda t: (-t[1], t[0])) if k >= 5
]

# ---- tally rows, n descending; solo row carries role/years per org ----
TYPE_RANK = {"company": 0, "university": 1, "lab": 2, "program": 3, "community": 4}  # by solo count desc

def org_entry(o: str) -> dict:
    org = ORG_BY_ID[o]
    e = {"id": o, "label": org["label"], "type": org["type"], "members": members[o]}
    if len(members[o]) == 1:
        l = link_of[(members[o][0], o)]
        e["role"] = l.get("role") or ""
        e["years"] = l.get("years") or ""
    return e

tally = []
for n in sorted(tally_counts, reverse=True):
    orgs = [o for o, ms in members.items() if len(ms) == n]
    if n == 1:
        orgs.sort(key=lambda o: (TYPE_RANK[ORG_BY_ID[o]["type"]], ORG_BY_ID[o]["label"].lower()))
    else:
        orgs.sort(key=lambda o: ORG_BY_ID[o]["label"].lower())
    tally.append({"n": n, "orgs": [org_entry(o) for o in orgs]})
assert [r["n"] for r in tally] == sorted({r["n"] for r in tally}, reverse=True)
assert tally[-1]["n"] == 1

_amb = ambassadors[0] if ambassadors else None
_amb_label = next((p["label"] for p in AFF["people"] if p["id"] == _amb["id"]), None) if _amb else None
headline = (
    f"<strong>{n_solo}</strong> of the {len(ORG_BY_ID)} organizations are held by exactly one "
    "person" + (f" — and {_amb_label} alone keeps <strong>{_amb['k']}</strong> embassies."
                if _amb_label else ".")
)

payload = {
    "slug": "embassies",
    "title": "Embassies",
    "headline": headline,
    "data": {
        "tally": tally,
        "solo_types": {t: solo_types[t] for t in sorted(solo_types, key=lambda t: TYPE_RANK[t])},
        "ambassadors": ambassadors,
        "holders": {p: holders[p] for p in sorted(holders)},
        "n_solo": n_solo,
        "n_orgs": len(ORG_BY_ID),
    },
}

blob = json.dumps(payload, separators=(",", ":"))
assert len(blob) < 300_000, f"payload too big: {len(blob)}"
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(blob)
print(f"[embassies] OK {len(blob)/1024:.0f}KB — {n_solo} of the {len(ORG_BY_ID)} organizations "
      f"are held by exactly one person"
      + (f" — and {_amb_label} alone keeps {_amb['k']} embassies." if _amb_label else "."))
