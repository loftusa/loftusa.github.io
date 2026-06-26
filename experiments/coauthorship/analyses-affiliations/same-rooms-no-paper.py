# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""same-rooms-no-paper — diff the rooms graph against the papers graph on the same 48 people.
Run: cd experiments/coauthorship/analyses-affiliations && uv run same-rooms-no-paper.py"""
import json
import re
from collections import Counter
from itertools import combinations
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
assert (REPO / "assets").exists(), f"REPO mis-resolved: {REPO}"

AFF = json.loads((REPO / "assets/data/affiliations.json").read_text())
GRAPH = json.loads((REPO / "assets/data/coauthorship.json").read_text())
OUT = REPO / "assets/data/analyses-affiliations" / "same-rooms-no-paper.json"

NOW = 2026  # fixed snapshot year — never datetime.now(), output must be byte-stable


def start_year(y):
    m = re.match(r"^\s*(\d{4})", y or "")
    return int(m.group(1)) if m else None


def parse_end(y):
    ys = re.findall(r"\d{4}", y or "")
    if not ys:
        return None
    return NOW if re.search(r"[–-]\s*$", y) else int(ys[-1])


norm = lambda s: re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()

# ---- the 48 people: AFF ids are already canonical graph node ids ----
people = [p["id"] for p in AFF["people"]]
assert len(people) >= 48, len(people)
pset = set(people)
gnode_ids = {n["id"] for n in GRAPH["nodes"]}
assert pset <= gnode_ids, pset - gnode_ids

org_label = {o["id"]: o["label"] for o in AFF["orgs"]}
org_ids = set(org_label)

# ---- layer 1: paper edges among the 48 (direct co-authorships) ----
edge_link: dict[tuple, dict] = {}
for l in GRAPH["links"]:
    a, b = norm(str(l["source"])), norm(str(l["target"]))
    if a in pset and b in pset:
        k = (a, b) if a < b else (b, a)
        assert k not in edge_link, f"duplicate edge {k}"
        edge_link[k] = l
assert edge_link
assert all(l["n_papers"] >= 1 and l["papers"] for l in edge_link.values())

# Verify completeness: every list-list pair co-appearing on any indexed paper is an edge.
# The one paper-derived pair NOT shipped is merullo x brinkmann — EDGE_DROP in build_graph.py
# (a 49-author proceedings volume, verified non-co-authorship), correctly absent here too.
papers = json.loads((REPO / "experiments/coauthorship/analyses/_derived/papers.json").read_text())
paper_pairs = set()
for p in papers:
    mem = sorted({m for m in p["members"] if m in pset})
    paper_pairs.update(combinations(mem, 2))
assert set(edge_link) - paper_pairs == set(), "shipped edge with no underlying paper"
assert paper_pairs - set(edge_link) == {("jack merullo", "jannik brinkmann")}, (
    "unexpected paper pair missing from the shipped graph"
)

# ---- layer 2: the affiliation projection (the rooms graph) ----
proj: dict[tuple, dict] = {}
for p in AFF["projection"]:
    a, b = p["a"], p["b"]
    assert a in pset and b in pset, (a, b)
    assert all(o in org_ids for o in p["shared"]), p["shared"]
    k = (a, b) if a < b else (b, a)
    assert k not in proj
    proj[k] = p
assert proj

# ---- the diff ----
open_pairs = sorted(set(proj) - set(edge_link))     # rooms shared, no paper
both_pairs = sorted(set(proj) & set(edge_link))     # rooms and paper
inv_pairs = sorted(set(edge_link) - set(proj))      # paper, no room
assert len(open_pairs) + len(both_pairs) == len(proj)
jaccard = len(both_pairs) / (len(open_pairs) + len(both_pairs) + len(inv_pairs))
assert 0 <= jaccard <= 1

# ---- per-pair overlap-years over shared orgs (only where BOTH links are dated) ----
years_of = {}
for l in AFF["links"]:
    k = (l["person"], l["org"])
    assert k not in years_of, f"duplicate person-org link {k}"
    years_of[k] = l["years"]


def overlap_years(a, b, shared_orgs):
    tot = 0
    for o in shared_orgs:
        sa, ea = start_year(years_of[(a, o)]), parse_end(years_of[(a, o)])
        sb, eb = start_year(years_of[(b, o)]), parse_end(years_of[(b, o)])
        if sa is None or ea is None or sb is None or eb is None:
            continue
        tot += max(0, min(ea, eb) - max(sa, sb))
    return tot


# ---- conversion bins by projection weight ----
open_hist = Counter(proj[k]["weight"] for k in open_pairs)
both_hist = Counter(proj[k]["weight"] for k in both_pairs)
assert sum(open_hist.values()) == len(open_pairs)
assert sum(both_hist.values()) == len(both_pairs)
BIN_LABEL = {  # dominant shared-room type per weight (type_weights in AFF["meta"])
    1.0: "community / campus",
    2.0: "program / company",
    3.0: "lab",
    5.0: "lab + company",
}
bins = [
    {"w": w, "open": open_hist.get(w, 0), "both": both_hist.get(w, 0), "label": BIN_LABEL[w]}
    for w in sorted(set(open_hist) | set(both_hist))
]
assert sum(b["open"] for b in bins) == 141 and sum(b["both"] for b in bins) == 72

# ---- all 182 connected pairs (open + both), the dot population ----
pairs = []
for a, b in sorted(open_pairs + both_pairs):
    p = proj[(a, b)]
    kind = "both" if (a, b) in edge_link else "open"
    pairs.append({
        "a": a, "b": b, "w": p["weight"], "kind": kind,
        "orgs": [org_label[o] for o in p["shared"]],
        "overlap": overlap_years(a, b, p["shared"]),
        "n_papers": edge_link[(a, b)]["n_papers"] if kind == "both" else 0,
    })
assert len(pairs) == len(proj)

# ---- inverse pairs: paper, no room ----
inverse = [
    {"a": a, "b": b, "n_papers": edge_link[(a, b)]["n_papers"],
     "papers": edge_link[(a, b)]["papers"][:3]}
    for a, b in inv_pairs
]
riders = sum(1 for i in inverse if "Open Problems in Mechanistic Interpretability" in i["papers"])
assert riders == 6, riders  # six of the ten ride one big position paper — stated as a caveat

# ---- nominations: open pairs where both people have an indexed record ----
no_papers = {n["id"] for n in GRAPH["nodes"] if n.get("no_papers")}
assert no_papers is not None
nameable = [(a, b) for a, b in open_pairs if a not in no_papers and b not in no_papers]
assert len(nameable) <= len(open_pairs)
assert sum(1 for a, b in open_pairs if a in no_papers or b in no_papers) == 13

ranked = sorted(
    nameable,
    key=lambda k: (-proj[k]["weight"], -overlap_years(*k, proj[k]["shared"]), k[0], k[1]),
)
nominations = [
    {"a": a, "b": b, "orgs": [org_label[o] for o in proj[(a, b)]["shared"]],
     "overlap": overlap_years(a, b, proj[(a, b)]["shared"])}
    for a, b in ranked[:8]
]

# ---- per-person open-pair counts for the minimap ----
open_count = Counter()
for a, b in open_pairs:
    open_count[a] += 1
    open_count[b] += 1
open_count_by_person = {k: open_count[k] for k in sorted(open_count)}
assert set(open_count_by_person) <= pset

# ---- envelope ----
headline = (
    f"<strong>{len(open_pairs)}</strong> pairs of people here have shared a lab, a program, or "
    f"an office without ever sharing a paper — and only <strong>{len(inv_pairs)}</strong> pairs "
    "managed the reverse, a paper with no room in common."
)
assert f"<strong>{len(open_pairs)}</strong>" in headline
assert f"<strong>{len(inv_pairs)}</strong>" in headline

payload = {
    "slug": "same-rooms-no-paper",
    "title": "Same rooms, no paper",
    "headline": headline,
    "data": {
        "counts": {
            "open": len(open_pairs), "both": len(both_pairs), "inverse": len(inv_pairs),
            "jaccard": round(jaccard, 3), "open_nameable": len(nameable),
        },
        "bins": bins,
        "pairs": pairs,
        "inverse": inverse,
        "nominations": nominations,
        "open_count_by_person": open_count_by_person,
    },
}

blob = json.dumps(payload, separators=(",", ":"))
assert len(blob) < 300_000, len(blob)
OUT.write_text(blob)
print(f"[same-rooms-no-paper] OK {len(blob)/1024:.0f}KB — " + re.sub(r"<[^>]+>", "", headline))
