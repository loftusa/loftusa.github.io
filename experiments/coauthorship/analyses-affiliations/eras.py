# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""eras — cohort waves per shared room (Lexis-flavored swimlanes over affiliation years).
Run: cd experiments/coauthorship/analyses-affiliations && uv run eras.py"""
import json
import re
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = Path(__file__).resolve().parents[3]  # repo root (worktree root)
assert (REPO / "public" / "assets").exists(), f"REPO mis-resolved: {REPO}"

AFF = json.loads((REPO / "public/assets/data/affiliations.json").read_text())
GRAPH = json.loads((REPO / "public/assets/data/coauthorship.json").read_text())
OUT = REPO / "public/assets/data/analyses-affiliations" / "eras.json"

GRAPH_IDS = {n["id"] for n in GRAPH["nodes"]}
ORG_IDS = {o["id"] for o in AFF["orgs"]}

# ---- year parsing (whole-year resolution; NOW is a constant, never datetime.now()) ----
NOW = 2026


def start_year(y):
    m = re.match(r"^\s*(\d{4})", y or "")
    return int(m.group(1)) if m else None


def parse_end(y):
    ys = re.findall(r"\d{4}", y or "")
    if not ys:
        return None
    return NOW if re.search(r"[–-]\s*$", y) else int(ys[-1])


# ---- drawn orgs: every room three or more people passed through ----
drawn = [o for o in AFF["orgs"] if o["n_members"] >= 3]
assert drawn, 'no orgs with n_members>=3'

links_by_org = defaultdict(list)
for l in AFF["links"]:
    if l["org"] in {o["id"] for o in drawn}:
        links_by_org[l["org"]].append(l)

orgs_out = []
n_undated_total = 0
for o in drawn:
    oid = o["id"]
    assert oid in ORG_IDS
    links = links_by_org[oid]
    assert len(links) == o["n_members"], (oid, len(links), o["n_members"])

    members, undated = [], []
    for l in links:
        pid = l["person"]
        assert pid in GRAPH_IDS, f"person not in graph: {pid}"
        years = l.get("years") or ""
        s, e = start_year(years), parse_end(years)
        ongoing = bool(re.search(r"[–-]\s*$", years))
        if s is None and e is None:
            undated.append(pid)
            continue
        spell = s == e  # a single calendar year: months-long stint, not a year bar
        if s is not None:
            assert 1950 <= s <= NOW, (pid, oid, years)
        if e is not None:
            assert 1950 <= e <= NOW, (pid, oid, years)
        if s is not None and e is not None:
            assert s <= e, (pid, oid, years)
        members.append(
            {
                "id": pid,
                "start": s,
                "end": e,
                "ongoing": ongoing,
                "spell": spell,
                "role": l.get("role") or "",
                "years": years,
            }
        )

    members.sort(key=lambda m: (m["start"] or 9999, m["end"] or 9999, m["id"]))
    undated.sort()
    n_undated_total += len(undated)

    starts = sorted({m["start"] for m in members if m["start"] is not None})
    assert starts, f"org with no dated start: {oid}"
    orgs_out.append(
        {
            "id": oid,
            "label": o["label"],
            "type": o["type"],
            "n": o["n_members"],
            "first": starts[0],
            "waves": len(starts),
            "members": members,
            "undated": undated,
        }
    )

orgs_out.sort(key=lambda g: (g["first"], g["id"]))

# ---- verified anchors (the headline's numbers live or die here) ----
by_id = {g["id"]: g for g in orgs_out}
assert min(g["first"] for g in orgs_out) >= 1990

assert sum(1 for o in AFF["orgs"] if o["n_members"] == 2) == 15  # the omitted two-member rooms

_top = sorted(orgs_out, key=lambda g: (-g["n"], g["first"]))[:3]
_top.sort(key=lambda g: g["first"])
_progs = [g for g in orgs_out if g["type"] == "program" and g["waves"] >= 3]
_w = max(_progs, key=lambda g: g["waves"]) if _progs else max(orgs_out, key=lambda g: g["waves"])
headline = (
    f"The chat arrived in waves: {_top[0]['label']} opens in <strong>{_top[0]['first']}</strong>, "
    + ", ".join(f"{g['label']} in {g['first']}" for g in _top[1:])
    + f" — and {_w['label']} has landed <strong>{_w['waves']}</strong> separate "
    + f"cohorts since {_w['first']}."
)

payload = {
    "slug": "eras",
    "title": "The eras",
    "headline": headline,
    "data": {"x": [min(g["first"] for g in orgs_out), NOW], "orgs": orgs_out},
}

blob = json.dumps(payload, separators=(",", ":"), ensure_ascii=False)
assert len(blob) < 300_000, len(blob)
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(blob)
print(f"[eras] OK {len(blob)/1024:.0f}KB — " + re.sub(r"</?strong>", "", headline))
