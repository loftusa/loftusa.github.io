# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""your-seat — the People capstone: a per-person digest gathered from the six panels.
Run: cd experiments/coauthorship/analyses-affiliations && uv run your-seat.py"""
import json
import re
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
assert (REPO / "assets").exists(), f"REPO mis-resolved: {REPO}"

AFF = json.loads((REPO / "assets/data/affiliations.json").read_text())
PANELS = REPO / "assets/data/analyses-affiliations"
SRP = json.loads((PANELS / "same-rooms-no-paper.json").read_text())["data"]
ERAS = json.loads((PANELS / "eras.json").read_text())["data"]
PIPE = json.loads((PANELS / "the-pipeline.json").read_text())["data"]
RANGE = json.loads((PANELS / "range.json").read_text())["data"]
EMB = json.loads((PANELS / "embassies.json").read_text())["data"]
WWN = json.loads((PANELS / "where-we-are-now.json").read_text())["data"]
SHARED = json.loads((PANELS / "shared.json").read_text())
OUT = PANELS / "your-seat.json"

NODES = set(SHARED["nodes"])
people = {p["id"]: p for p in AFF["people"]}
assert len(people) >= 48, len(people)
assert set(people) <= NODES
assert len({p["label"] for p in people.values()}) == len(people)  # labels unique → picker can key on them
for pid, p in people.items():
    assert p["label"] == SHARED["nodes"][pid]["label"], pid
assert "alex loftus" in people  # the JS default seat

orgs = {o["id"]: o for o in AFF["orgs"]}
assert len({o["label"] for o in orgs.values()}) == len(orgs)
org_label = {oid: o["label"] for oid, o in orgs.items()}
# label -> type, for the JS type-coloring of org names (ties + embassies ship labels only)
org_types = {o["label"]: o["type"] for o in sorted(orgs.values(), key=lambda o: o["label"])}

# ---- year parsing — same start/end regex spec as eras.py (kept in sync by hand) ----
NOW = 2026


def start_year(y):
    m = re.match(r"^\s*(\d{4})", y or "")
    return int(m.group(1)) if m else None


def parse_end(y):
    ys = re.findall(r"\d{4}", y or "")
    if not ys:
        return None
    return NOW if re.search(r"[–-]\s*$", y) else int(ys[-1])


# ---- ties: the nesting-discounted projection, read per person ------------------------------
ties_by: dict[str, list[dict]] = defaultdict(list)
for e in AFF["projection"]:
    assert e["a"] in NODES and e["b"] in NODES, (e["a"], e["b"])
    shared_labels = [org_label[s] for s in e["shared"]]
    for me, other in ((e["a"], e["b"]), (e["b"], e["a"])):
        ties_by[me].append({"other": other, "w": e["weight"], "orgs": shared_labels})
for ls in ties_by.values():
    ls.sort(key=lambda t: (-t["w"], t["other"]))

# ---- invitations: kind=="open" pairs from same-rooms-no-paper ------------------------------
inv_by: dict[str, list[dict]] = defaultdict(list)
for pr in SRP["pairs"]:
    if pr["kind"] != "open":
        continue
    assert pr["a"] in NODES and pr["b"] in NODES
    assert all(o in org_types for o in pr["orgs"]), pr["orgs"]
    for me, other in ((pr["a"], pr["b"]), (pr["b"], pr["a"])):
        inv_by[me].append({"other": other, "orgs": pr["orgs"], "overlap": pr["overlap"]})
for ls in inv_by.values():
    ls.sort(key=lambda v: (-v["overlap"], v["other"]))
for pid in people:  # published per-person open counts must agree with the pair list
    assert SRP["open_count_by_person"].get(pid, 0) == len(inv_by.get(pid, [])), pid

# ---- chapters: eras spans where drawn, supplemented from raw links everywhere else ---------
era_member = {}  # (pid, oid) -> (org record, member record)
for o in ERAS["orgs"]:
    for m in o["members"]:
        era_member[(m["id"], o["id"])] = (o, m)

links_by_person: dict[str, list[dict]] = defaultdict(list)
seen = set()
for l in AFF["links"]:
    key = (l["person"], l["org"])
    assert key not in seen, key  # one chapter per (person, org)
    seen.add(key)
    assert l["org"] in orgs and l["person"] in people
    links_by_person[l["person"]].append(l)
assert all(k in seen for k in era_member), "eras member without a backing link"

chapters_of: dict[str, list[dict]] = {}
for pid in sorted(people):
    ch = []
    for l in links_by_person[pid]:
        oid = l["org"]
        if (pid, oid) in era_member:  # drawn org: reuse the eras record verbatim
            o, m = era_member[(pid, oid)]
            ch.append({"org": o["label"], "type": o["type"], "start": m["start"],
                       "end": m["end"], "ongoing": m["ongoing"], "spell": m["spell"],
                       "role": m["role"]})
        else:  # < 3-member org, or undated member of a drawn org
            years = l.get("years") or ""
            s, e = start_year(years), parse_end(years)
            ch.append({"org": org_label[oid], "type": orgs[oid]["type"], "start": s,
                       "end": e, "ongoing": bool(re.search(r"[–-]\s*$", years)),
                       "spell": s is not None and s == e, "role": l.get("role") or ""})
    ch.sort(key=lambda c: (c["start"] if c["start"] is not None else 9999,
                           c["end"] if c["end"] is not None else 9999, c["org"]))
    chapters_of[pid] = ch
assert sum(map(len, chapters_of.values())) == len(AFF["links"]) == 237

# ---- pipeline: dated type sequence in career order; next from the published probs row ------
TYPES = PIPE["types"]
assert len(TYPES) == 5 and all(len(r) == 5 for r in PIPE["probs"])
dated_by: dict[str, list[dict]] = defaultdict(list)
for l in AFF["links"]:
    if start_year(l.get("years")) is not None:
        dated_by[l["person"]].append(l)
for ls in dated_by.values():  # same career order as the-pipeline.py: (start year, org id)
    ls.sort(key=lambda l: (start_year(l["years"]), l["org"]))
assert set(dated_by) == set(PIPE["latest_type"])  # 45 people; 3 have no dated chapters


def next_two(cur: str) -> list[dict]:
    row = PIPE["probs"][TYPES.index(cur)]
    ranked = sorted(zip(TYPES, row), key=lambda tp: (-tp[1], TYPES.index(tp[0])))
    return [{"type": t, "p": p} for t, p in ranked[:2]]


# ---- range: published scores + the quadrant word -------------------------------------------
rng_by = {p["id"]: p for p in RANGE["people"]}
assert set(rng_by) == set(people)


def quadrant(r: dict) -> str | None:
    if r["n1"] >= 3.5:
        return "wide traveler"
    if r["n"] <= 4 and r["n1"] <= 1.5 and not r["low"]:
        return "deep anchor"
    return None


# ---- embassies: solo-org labels per holder --------------------------------------------------
solo_rows = [t for t in EMB["tally"] if t["n"] == 1]
assert len(solo_rows) == 1 and len(solo_rows[0]["orgs"]) == EMB["n_solo"] == 125
solo_by: dict[str, list[str]] = defaultdict(list)
for o in solo_rows[0]["orgs"]:
    assert len(o["members"]) == 1 and o["label"] in org_types
    solo_by[o["members"][0]].append(o["label"])
for ls in solo_by.values():
    ls.sort()
assert {p: len(v) for p, v in solo_by.items()} == EMB["holders"]

# ---- now: the census row -------------------------------------------------------------------
wwn_by = {p["id"]: p for p in WWN["people"]}
assert set(wwn_by) == set(people)
no_cur = set(WWN["no_current_ids"])
assert all((not wwn_by[pid]["all_current"]) == (pid in no_cur) for pid in people)

# ---- assemble digests ------------------------------------------------------------------------
digests: dict[str, dict] = {}
for pid in sorted(people):
    t_all = ties_by.get(pid, [])
    inv_all = inv_by.get(pid, [])
    w, r = wwn_by[pid], rng_by[pid]
    solo = solo_by.get(pid, [])
    dig = {
        "ties": t_all[:5],
        "n_ties": len(t_all),
        "tie_ids": sorted(t["other"] for t in t_all),  # full ego set for the minimap
        "invitations": inv_all[:6],
        "n_invitations": len(inv_all),
        "chapters": chapters_of[pid],
        "range": {"n": r["n"], "n1": r["n1"], "low": r["low"], "quadrant": quadrant(r)},
        "embassies": {"orgs": solo[:8], "n": len(solo)},
        "now": {"org": w["all_current"][0]["org"] if w["all_current"] else None,
                "city": w["city"] or None,
                "no_current": pid in no_cur},
        "sparse": not t_all and not inv_all,
    }
    if pid in dated_by:  # omitted entirely when no dated chapters exist
        seq = [orgs[l["org"]]["type"] for l in dated_by[pid]]
        assert seq[-1] == PIPE["latest_type"][pid], pid
        dig["pipeline"] = {"seq": seq, "current": seq[-1], "next": next_two(seq[-1])}
    digests[pid] = dig

sparse_ids = sorted(pid for pid, dd in digests.items() if dd["sparse"])
shipped_others = {t["other"] for dd in digests.values() for t in dd["ties"]} | \
    {v["other"] for dd in digests.values() for v in dd["invitations"]} | \
    {i for dd in digests.values() for i in dd["tie_ids"]}
assert shipped_others <= NODES

order = sorted(people, key=lambda pid: (people[pid]["label"], pid))
assert len(order) == len(people)

headline = (f"There are <strong>{len(people)}</strong> seats at this table — "
            "pick a name and see what the network holds for them.")

payload = {
    "slug": "your-seat",
    "title": "Your seat",
    "headline": headline,
    "data": {"order": order, "people": digests, "org_types": org_types},
}
blob = json.dumps(payload, separators=(",", ":"))
assert len(blob) < 300_000, len(blob)
OUT.write_text(blob)
print(f"[your-seat] OK {len(blob)/1024:.0f}KB — " + re.sub(r"</?strong>", "", headline))
