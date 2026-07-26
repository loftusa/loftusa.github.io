# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""where-we-are-now — cross-sectional census: who's where today, and who's near whom.
Run: cd experiments/coauthorship/analyses-affiliations && uv run where-we-are-now.py"""
import importlib.util
import json
import re
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
assert (REPO / "public" / "assets").exists(), f"REPO mis-resolved: {REPO}"

AFF = json.loads((REPO / "public/assets/data/affiliations.json").read_text())
GRAPH = json.loads((REPO / "public/assets/data/coauthorship.json").read_text())
OUT = REPO / "public/assets/data/analyses-affiliations" / "where-we-are-now.json"

# org canon (CANON/DROP/slug) imported read-only from the build script — never copied
_spec = importlib.util.spec_from_file_location(
    "build_affiliations", REPO / "experiments/coauthorship/build_affiliations.py")
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
SRC = _mod.load_src()      # the source WITH the self-service overlay (current flags live here)
CANON, DROP, slug = _mod.CANON, _mod.DROP, _mod.slug

# ---- join keys: SRC names -> canonical graph node ids -------------------------------------
norm = _mod.norm                      # the build's join key — never copied
graph_ids = {n["id"] for n in GRAPH["nodes"]}
aff_people = {p["id"]: p for p in AFF["people"]}
org_ids = {o["id"] for o in AFF["orgs"]}
pid_of = {k: norm(k) for k in SRC}
assert len(SRC) == len(aff_people) and len(set(pid_of.values())) == len(SRC)
for pid in pid_of.values():
    assert pid in aff_people, f"SRC person not in AFF: {pid}"
    assert pid in graph_ids, f"person id not a graph node: {pid}"

# ---- current affiliations, canonicalized, deduped per person ------------------------------
n_current_raw = sum(1 for v in SRC.values() for e in v["entries"] if e["current"])
assert n_current_raw > 0

cur_of: dict[str, dict[str, dict]] = {}            # pid -> oid -> {org,type,years}
for name, rec in SRC.items():
    pid = pid_of[name]
    per_org: dict[str, dict] = {}
    for e in rec["entries"]:
        if not e["current"] or e["org"] in DROP:
            continue
        label, typ = CANON.get(e["org"], (e["org"], e["type"]))
        oid = slug(label)
        assert oid in org_ids, f"current org not in AFF orgs: {oid}"
        if oid in per_org:                          # same canonical org twice (fred's Harvard schools)
            assert per_org[oid]["type"] == typ
            per_org[oid]["_years"].append((e.get("years", ""), True))
        else:
            per_org[oid] = {"org": label, "type": typ, "_years": [(e.get("years", ""), True)]}
    for m in per_org.values():
        m["years"] = _mod.normalize_years(m.pop("_years"))   # same fold rule as the shipped links
    cur_of[pid] = per_org

# ---- people-level type counts, the split, primary assignment -------------------------------
PREC = ["company", "lab", "university", "program", "community"]
types_of = {pid: {m["type"] for m in per.values()} for pid, per in cur_of.items()}

no_current = sorted(pid for pid, ts in types_of.items() if not ts)
assert len(no_current) < len(SRC) // 2   # most people have a current chapter

type_counts_people = Counter(t for ts in types_of.values() for t in ts)

company_now = {pid for pid, ts in types_of.items() if "company" in ts}
academia_now = {pid for pid, ts in types_of.items() if ts & {"university", "lab"}}
straddlers = sorted(company_now & academia_now)
split = {"company_now": len(company_now), "academia_now": len(academia_now),
         "both": len(straddlers)}
assert split["both"] <= min(split["company_now"], split["academia_now"])

primary_of = {pid: next((t for t in PREC if t in ts), None) for pid, ts in types_of.items()}
assigned_counts = Counter(t for t in primary_of.values() if t)
assert sum(assigned_counts.values()) == len(SRC) - len(no_current)
assert sum(assigned_counts.values()) == 50     # 54 people minus the 4 with no current chapter

# ---- cities: fold variants, group metros ---------------------------------------------------
CITY_CANON = {"Cambridge, Massachusetts": "Cambridge, MA",
              "Seattle, Washington (area)": "Seattle",
              "Brooklyn, NY (Williamsburg)": "New York",
              "Berkeley, CA": "Berkeley"}
METRO = {"Greater Boston": {"Boston", "Cambridge, MA"},
         "Bay Area": {"San Francisco", "Berkeley", "Stanford"}}
metro_of = lambda c: next((m for m, s in METRO.items() if c in s), c)

city_of = {pid_of[k]: CITY_CANON.get(v.get("city", ""), v.get("city", ""))
           for k, v in SRC.items()}
unknown = sorted(pid for pid, c in city_of.items() if not c)
assert len(unknown) < len(SRC)

city_members: dict[str, list[str]] = {}
for pid in sorted(city_of):
    if city_of[pid]:
        city_members.setdefault(city_of[pid], []).append(pid)
metro_size = Counter()
for c, mem in city_members.items():
    metro_size[metro_of(c)] += len(mem)
assert sum(1 for n in metro_size.values() if n == 1) == 8      # Zurich (Antonio Mari) joins as a new singleton
assert sum(metro_size.values()) + len(unknown) == 54

# rows ordered: metro size desc, metro name; sub-cities by size desc, name
comm = lambda pid: aff_people[pid]["community"]
cities = []
for metro in sorted(metro_size, key=lambda m: (-metro_size[m], m)):
    subs = [c for c in city_members if metro_of(c) == metro]
    for c in sorted(subs, key=lambda c: (-len(city_members[c]), c)):
        members = sorted(city_members[c], key=lambda pid: (comm(pid), pid))
        cities.append({"name": c, "metro": metro, "members": members})

# ---- payload -------------------------------------------------------------------------------
people = []
for pid in sorted(cur_of):
    per = cur_of[pid]
    all_current = sorted(per.values(), key=lambda m: (PREC.index(m["type"]), m["org"]))
    people.append({
        "id": pid,
        "initials": aff_people[pid]["initials"],
        "primary": primary_of[pid],
        "all_current": all_current,
        "straddler": pid in straddlers,
        "city": city_of[pid],
        "metro": metro_of(city_of[pid]) if city_of[pid] else "",
    })
assert len(people) == len(SRC) and all(p["id"] in graph_ids for p in people)
assert sum(len(p["all_current"]) for p in people) == 91        # raw current hats, fred's Harvard folded

gb = metro_size["Greater Boston"]
_even = abs(split["company_now"] - split["academia_now"]) <= 6
opener = ("Today the group splits almost exactly down the middle — " if _even
          else "Today the group splits ")
boston = (f" — and {gb} people are one subway ride apart in Greater Boston" if gb >= 3 else "")
headline = (opener
            + f"<strong>{split['company_now']}</strong> people inside companies, "
            + f"<strong>{split['academia_now']}</strong> in universities or labs, "
            + f"{split['both']} doing both" + boston + ".")

payload = {
    "slug": "where-we-are-now",
    "title": "Where everyone is now",
    "headline": headline,
    "data": {
        "people": people,
        "type_counts_people": {t: type_counts_people[t] for t in PREC},
        "split": split,
        "assigned_counts": {t: assigned_counts.get(t, 0) for t in PREC},
        "no_current": len(no_current),
        "no_current_ids": no_current,
        "cities": cities,
        "n_city_unknown": len(unknown),
    },
}
blob = json.dumps(payload, separators=(",", ":"))
assert len(blob) < 300_000
OUT.write_text(blob)
print(f"[where-we-are-now] OK {len(blob)/1024:.0f}KB — " + re.sub(r"</?strong>", "", headline))
