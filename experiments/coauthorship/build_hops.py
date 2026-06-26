# /// script
# requires-python = ">=3.10"
# ///
"""Build the affiliation hop layer: people OUTSIDE the map who verifiably shared a room
with a member — so the careers map's reach slider can reveal "who you know in common",
mirroring the papers map's hop reveal.

An outside person earns a link to a mapped org only through a CO-EVENT, never a shared
label alone: source "openalex" means they co-authored a paper with a member while both
listed that institution on the paper (room + moment + artifact). Other sources (github,
orcid) merge from hop_sources/*.json when present, with the same link schema.

    cd experiments/coauthorship && uv run build_hops.py

Reads:  ../../assets/data/affiliations.json   (canonical orgs + member ids — build first)
        seeds.json                            (member -> trusted OpenAlex author ids)
        raw/oa_works_v2_*.json                (cached works; no network calls here)
        hop_sources/*.json                    (optional extra sources, same link schema)
Writes: ../../assets/data/affiliations-hops.json. Deterministic: byte-identical on re-run.
"""
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_affiliations import CANON, slug  # noqa: E402  (one org-identity layer, two consumers)

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
AFF = REPO / "assets" / "data" / "affiliations.json"
SEEDS = HERE / "seeds.json"
RAW = HERE / "raw"
SOURCES_DIR = HERE / "hop_sources"
OUT = REPO / "assets" / "data" / "affiliations-hops.json"
INDEX_OUT = REPO / "assets" / "data" / "guest-index.json"

INDEX_CAP = 300          # most-collaborating outside coauthors shipped to the finder
OA_TYPE = {"education": "university", "company": "company", "facility": "lab",
           "government": "lab", "healthcare": "university", "nonprofit": "community"}

FANOUT_CAP = 10          # per (member, org): keep the top-K outside people by co-event count
COUNTRYISH = (r"\s*\((United States|United Kingdom|Canada|France|Germany|Switzerland|Israel|"
              r"China|Japan|Australia|Netherlands|Sweden|Belgium|India|Italy|Spain|Austria)\)$")

ni = lambda s: re.sub(r"[^a-z0-9]+", " ", str(s).lower()).strip()   # mirrors the build's person norm
# identity key for OUTSIDE people: middle initials dropped, so "Joshua T Vogelstein" and
# "Joshua Vogelstein" (two OA author ids, one human) fold together — and can't slip past
# the member guard on a stray initial
name_key = lambda s: " ".join(t for t in ni(s).split() if len(t) > 1)


def make_inst_resolver(org_labels: dict[str, str]):
    """institution display_name -> mapped org id, or None. CANON first (the hand-curated
    identity layer), then the country-suffix-stripped name, then an exact label match."""
    def resolve(name: str) -> str | None:
        if not name:
            return None
        for cand in (name, re.sub(COUNTRYISH, "", name)):
            if cand in CANON:
                return slug(CANON[cand][0])
            if cand.lower() in org_labels:
                return org_labels[cand.lower()]
        return None
    return resolve


def fold_same_person(links: dict, names: dict) -> tuple[dict, dict]:
    """One human, several OA author ids: fold link evidence onto the smallest id per name."""
    canon: dict[str, str] = {}
    for aid in sorted(names):
        canon.setdefault(name_key(names[aid]), aid)
    out_links: dict[tuple[str, str], dict] = {}
    for (aid, org), ev in links.items():
        cid = canon[name_key(names[aid])]
        tgt = out_links.setdefault((cid, org), {"via": set(), "works": [], "years": []})
        tgt["via"] |= ev["via"]
        tgt["works"] += ev["works"]
        tgt["years"] += ev["years"]
    return out_links, {aid: names[aid] for aid in set(canon.values())}


def openalex_links(member_ids: set[str], org_labels: dict[str, str]) -> tuple[dict, dict]:
    """(outside_id, org_id) -> evidence, plus outside_id -> display name. Co-event only:
    the outside author and a member listed the same mapped institution on the same work."""
    seeds = json.loads(SEEDS.read_text())
    oa2member: dict[str, str] = {}
    for e in seeds:
        for i in [e.get("oa_id")] + e.get("oa_merge_ids", []):
            if i:
                oa2member[i.rstrip("/").split("/")[-1]] = ni(e["name"])
    member_keys = {name_key(m) for m in member_ids}
    resolve = make_inst_resolver(org_labels)
    bare = lambda a: ((a.get("author") or {}).get("id") or "").rstrip("/").split("/")[-1]

    links: dict[tuple[str, str], dict] = {}
    names: dict[str, str] = {}
    seen_works: set[str] = set()
    for f in sorted(RAW.glob("oa_works_v2_*.json")):
        for w in json.loads(f.read_text()).get("results", []):
            if w["id"] in seen_works:
                continue
            seen_works.add(w["id"])
            ash = w.get("authorships", [])
            members = [(a, oa2member[bare(a)]) for a in ash if bare(a) in oa2member]
            if not members:
                continue
            for a in ash:
                aid = bare(a)
                nm = (a.get("author") or {}).get("display_name") or a.get("raw_author_name") or ""
                # not an outside person if their OA id OR their name belongs to a member
                # (members sometimes surface under a second, unmerged OA id)
                if not aid or aid in oa2member or name_key(nm) in member_keys:
                    continue
                a_orgs = {o for i in a.get("institutions", []) if (o := resolve(i.get("display_name")))}
                if not a_orgs:
                    continue
                for ma, mname in members:
                    shared = a_orgs & {o for i in ma.get("institutions", [])
                                       if (o := resolve(i.get("display_name")))}
                    for org in shared:
                        ev = links.setdefault((aid, org), {"via": set(), "works": [], "years": []})
                        ev["via"].add(mname)
                        ev["works"].append((w.get("publication_year") or 0, w.get("title") or ""))
                        if w.get("publication_year"):
                            ev["years"].append(w["publication_year"])
                        names[aid] = nm
    return fold_same_person(links, names)


def build_guest_index(member_ids: set[str], orgs_by_id: dict[str, dict],
                      org_labels: dict[str, str]) -> list[dict]:
    """The extended social graph, ranked for the finder: every OUTSIDE coauthor in the works
    cache, ordered by how many papers they share with members — the people most likely to be
    searched for. Each ships ready-made career entries (resolver-canonicalized) so the map can
    place them instantly, no API round-trip."""
    seeds = json.loads(SEEDS.read_text())
    oa2member: dict[str, str] = {}
    for e in seeds:
        for i in [e.get("oa_id")] + e.get("oa_merge_ids", []):
            if i:
                oa2member[i.rstrip("/").split("/")[-1]] = ni(e["name"])
    member_keys = {name_key(m) for m in member_ids}
    resolve = make_inst_resolver(org_labels)
    bare = lambda a: ((a.get("author") or {}).get("id") or "").rstrip("/").split("/")[-1]

    acc: dict[str, dict] = {}                   # outside OA id -> {name, n, via, insts}
    seen_works: set[str] = set()
    for f in sorted(RAW.glob("oa_works_v2_*.json")):
        for w in json.loads(f.read_text()).get("results", []):
            if w["id"] in seen_works:
                continue
            seen_works.add(w["id"])
            ash = w.get("authorships", [])
            members = sorted({oa2member[bare(a)] for a in ash if bare(a) in oa2member})
            if not members:
                continue
            yr = w.get("publication_year")
            for a in ash:
                aid = bare(a)
                nm = (a.get("author") or {}).get("display_name") or a.get("raw_author_name") or ""
                if not aid or aid in oa2member or name_key(nm) in member_keys:
                    continue
                rec = acc.setdefault(aid, {"name": nm, "n": 0, "via": set(), "insts": {}})
                rec["n"] += 1
                rec["via"] |= set(members)
                for inst in a.get("institutions", []):
                    iname = inst.get("display_name")
                    if not iname:
                        continue
                    slot = rec["insts"].setdefault(iname, {"type": inst.get("type") or "", "years": set()})
                    if yr:
                        slot["years"].add(yr)

    # one human across several OA ids: keep the record with the most co-papers per name
    by_name: dict[str, tuple[str, dict]] = {}
    for aid, rec in sorted(acc.items()):
        k = name_key(rec["name"])
        if k and (k not in by_name or rec["n"] > by_name[k][1]["n"]):
            by_name[k] = (aid, rec)
    ranked = sorted(by_name.values(), key=lambda t: (-t[1]["n"], t[1]["name"].lower(), t[0]))[:INDEX_CAP]

    index = []
    for aid, rec in ranked:
        entries = []
        for iname, slot in sorted(rec["insts"].items(), key=lambda kv: -len(kv[1]["years"])):
            oid = resolve(iname)
            ys = sorted(slot["years"])
            entries.append({
                "org": orgs_by_id[oid]["label"] if oid else re.sub(COUNTRYISH, "", iname),
                "type": orgs_by_id[oid]["type"] if oid else OA_TYPE.get(slot["type"], "community"),
                "years": (f"{ys[0]}–{ys[-1]}" if ys[0] != ys[-1] else str(ys[0])) if ys else "",
            })
        index.append({"label": rec["name"], "oa": aid, "n": rec["n"],
                      "via": sorted(rec["via"])[:4], "entries": entries[:8]})
    return index


def main() -> None:
    aff = json.loads(AFF.read_text())
    member_ids = {p["id"] for p in aff["people"]}
    org_ids = {o["id"] for o in aff["orgs"]}
    org_labels = {o["label"].lower(): o["id"] for o in aff["orgs"]}

    raw_links, names = openalex_links(member_ids, org_labels)

    # fan-out cap: for each (member, org) room, keep only the top-K outside people by
    # co-event count (ties by name) — the slider should reveal the strongest hops, not 114
    keep: set[tuple[str, str]] = set()
    by_member_org: dict[tuple[str, str], list] = defaultdict(list)
    for (aid, org), ev in raw_links.items():
        for m in ev["via"]:
            by_member_org[(m, org)].append((-len(ev["works"]), names[aid], aid))
    for (m, org), cands in by_member_org.items():
        for _, _, aid in sorted(cands)[:FANOUT_CAP]:
            keep.add((aid, org))

    initials = lambda s: "".join(w[0] for w in s.split()[:2]).upper() or "?"
    out_links = []
    used: set[str] = set()
    for (aid, org) in sorted(keep, key=lambda k: (names[k[0]].lower(), k[1])):
        ev = raw_links[(aid, org)]
        yrs = sorted(ev["years"])
        top_year, top_title = max(ev["works"])
        out_links.append({
            "person": "h:" + aid, "org": org,
            "via": sorted(ev["via"]),
            "n": len(ev["works"]),
            "years": (f"{yrs[0]}–{yrs[-1]}" if yrs[0] != yrs[-1] else str(yrs[0])) if yrs else "",
            "top": f"{top_title} ({top_year})" if top_year else top_title,
            "src": "openalex",
        })
        used.add(aid)
    people = [{"id": "h:" + aid, "label": names[aid], "initials": initials(names[aid])}
              for aid in sorted(used, key=lambda a: (names[a].lower(), a))]

    # optional extra sources (github, orcid …) — same shapes; the same human under two
    # sources (or a member under an alt account) is folded/skipped by name
    member_keys = {name_key(m) for m in member_ids}
    for f in sorted(SOURCES_DIR.glob("*.json")) if SOURCES_DIR.exists() else []:
        extra = json.loads(f.read_text())
        have_ids = {p["id"] for p in people}
        have_names = {name_key(p["label"]) for p in people} | member_keys
        fresh = [p for p in extra.get("people", [])
                 if p["id"] not in have_ids and name_key(p["label"]) not in have_names]
        fresh_ids = {p["id"] for p in fresh}
        people += fresh
        out_links += [l for l in extra.get("links", []) if l["person"] in fresh_ids]

    # fail-fast invariants
    pid_set = {p["id"] for p in people}
    assert len(pid_set) == len(people), "duplicate hop person id"
    assert not (pid_set & member_ids) and all(p["id"].startswith("h:") for p in people)
    for l in out_links:
        assert l["org"] in org_ids and l["person"] in pid_set, l
        assert l["via"] and all(v in member_ids for v in l["via"]), l
        assert l["n"] >= 1 and l["src"], l
    assert all(name_key(p["label"]) not in member_keys for p in people), "member leaked into hop layer"
    keys = [name_key(p["label"]) for p in people]
    assert len(set(keys)) == len(keys), "same human twice in the hop layer"

    # institution-name -> org-id table for the page's guest search (the JS mirror of
    # make_inst_resolver: lowercase lookup after stripping the country suffix client-side)
    inst2org = {k.lower(): slug(v[0]) for k, v in CANON.items() if slug(v[0]) in org_ids}
    inst2org.update({lbl: oid for lbl, oid in org_labels.items()})
    out = {"people": people, "links": out_links, "meta": {"inst2org": inst2org}}
    OUT.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")

    orgs_by_id = {o["id"]: o for o in aff["orgs"]}
    index = build_guest_index(member_ids, orgs_by_id, org_labels)
    assert all(name_key(p["label"]) not in {name_key(m) for m in member_ids} for p in index)
    INDEX_OUT.write_text(json.dumps({"people": index}, indent=1, ensure_ascii=False) + "\n")
    print(f"wrote {INDEX_OUT.relative_to(REPO)}: {len(index)} extended-graph people "
          f"(top: {', '.join(p['label'] for p in index[:5])})")
    by_src = defaultdict(int)
    for l in out_links:
        by_src[l["src"]] += 1
    print(f"wrote {OUT.relative_to(REPO)}: {len(people)} outside people, "
          f"{len(out_links)} links ({dict(by_src)}; capped from {len(raw_links)})")


if __name__ == "__main__":
    main()
