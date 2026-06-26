# /// script
# requires-python = ">=3.10"
# ///
"""Build the affiliation-graph JSON for /coauthorship/affiliations/ from affiliations.json.

The sweep wrote each person's affiliations with free-text org names, so one real org appears
under many spellings ("Bau Lab (David Bau's...)", "Northeastern University (David Bau's lab /
NDIF)", ...). CANON maps every variant to one canonical (label, type) — this file is the
hand-curated org identity layer, the analogue of seeds.json for the co-authorship graph.
People keep the map page's identity (label, community colour) by joining on coauthorship.json.

    cd experiments/coauthorship && uv run build_affiliations.py

Output: ../../assets/data/affiliations.json — people, orgs, person→org links (role/years/source
kept for the UI), and a person–person projection (type-weighted shared-org ties). Deterministic:
running twice must be byte-identical.
"""
import json
import re
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))   # sibling import works under importlib too
from affiliation_events import (ENTRY_TYPES, apply_aff_overlay,  # noqa: E402
                                load_aff_overrides, norm_person as norm)
from registry import ROSTER_PATH, reconcile_membership  # noqa: E402  (membership contract)

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
SRC = HERE / "affiliations.json"
OVERRIDES = HERE / "affiliation_overrides.json"   # folded self-service events (merge_affiliations.py)
GRAPH = REPO / "assets" / "data" / "coauthorship.json"
OUT = REPO / "assets" / "data" / "affiliations.json"
SHARED_OUT = REPO / "assets" / "data" / "analyses-affiliations" / "shared.json"

# how strongly a shared org ties two people in the person–person projection
TYPE_WEIGHT = {"lab": 3.0, "program": 2.0, "company": 2.0, "community": 1.0, "university": 1.0}
assert set(TYPE_WEIGHT) == ENTRY_TYPES   # one vocabulary, two homes — fail loud on drift

# org string (exact, as in affiliations.json) -> (canonical label, canonical type).
# Anything not listed keeps its own (org, type). Keyed by string only: where the same string
# appears under two types (Kaggle, Johns Hopkins University), the canon type wins for both.
CANON: dict[str, tuple[str, str]] = {
    # --- Bau lab (7 variants) ---
    "Bau Lab (David Bau's interpretability lab, Northeastern / NDIF)": ("Bau Lab (Northeastern / NDIF)", "lab"),
    "David Bau's Interpretable Neural Networks Lab (Bau Lab, Northeastern)": ("Bau Lab (Northeastern / NDIF)", "lab"),
    "David Bau's lab (Baulab, Northeastern interpretability lab)": ("Bau Lab (Northeastern / NDIF)", "lab"),
    "NDIF (National Deep Inference Fabric), David Bau's lab, Northeastern University": ("Bau Lab (Northeastern / NDIF)", "lab"),
    "Northeastern University (David Bau's Interpretable Neural Networks Lab)": ("Bau Lab (Northeastern / NDIF)", "lab"),
    "Northeastern University (David Bau's Interpretable Neural Networks lab)": ("Bau Lab (Northeastern / NDIF)", "lab"),
    "Northeastern University (David Bau's lab / NDIF)": ("Bau Lab (Northeastern / NDIF)", "lab"),
    # --- NeuroData / Vogelstein-Priebe circle at JHU (10 variants) ---
    "NeuroData Lab, Johns Hopkins University": ("NeuroData Lab (Johns Hopkins)", "lab"),
    "NeuroData lab, Johns Hopkins University": ("NeuroData Lab (Johns Hopkins)", "lab"),
    "NeuroData (Vogelstein lab)": ("NeuroData Lab (Johns Hopkins)", "lab"),
    "NeuroData (Vogelstein/Priebe group), Johns Hopkins University": ("NeuroData Lab (Johns Hopkins)", "lab"),
    "NeuroData lab (Johns Hopkins Center for Imaging Science)": ("NeuroData Lab (Johns Hopkins)", "lab"),
    "NeuroData lab (Johns Hopkins, Joshua Vogelstein / Carey Priebe)": ("NeuroData Lab (Johns Hopkins)", "lab"),
    "NeuroData lab (Johns Hopkins, Vogelstein/Miller)": ("NeuroData Lab (Johns Hopkins)", "lab"),
    "NeuroData lab (Joshua Vogelstein, Johns Hopkins University)": ("NeuroData Lab (Johns Hopkins)", "lab"),
    "NeuroData Laboratory (Vogelstein lab, JHU Biomedical Engineering)": ("NeuroData Lab (Johns Hopkins)", "lab"),
    "JHU NeuroData / Center for Imaging Science (Vogelstein–Priebe circle)": ("NeuroData Lab (Johns Hopkins)", "lab"),
    # --- JHU umbrella (brandon's generic [lab] entry folds into the university node) ---
    "Johns Hopkins University": ("Johns Hopkins University", "university"),
    "Johns Hopkins University Applied Physics Laboratory": ("JHU Applied Physics Laboratory", "lab"),
    "Mathematical Institute for Data Science (MINDS), Johns Hopkins": ("MINDS (Johns Hopkins)", "lab"),
    "Mao Lab (Mao Group), Johns Hopkins University": ("Mao Lab (Johns Hopkins)", "lab"),
    "Saria Lab, Johns Hopkins University": ("Saria Lab (Johns Hopkins)", "lab"),
    # --- programs with redundant parentheticals ---
    "MATS (ML Alignment & Theory Scholars)": ("MATS", "program"),
    "ARENA (Alignment Research Engineer Accelerator)": ("ARENA", "program"),
    "SPAR (Supervised Program for Alignment Research)": ("SPAR", "program"),
    "ML4G (Machine Learning for Good AI-safety bootcamp)": ("ML4G", "program"),
    "Centre for the Governance of AI (GovAI)": ("GovAI", "program"),
    "AI Safety Camp": ("AI Safety Camp", "program"),
    "AI Safety Camp (AISC 2024)": ("AI Safety Camp", "program"),
    "AI Safety Camp (AISC8)": ("AI Safety Camp", "program"),
    "LASR Labs (London AI Safety Research)": ("LASR Labs", "program"),
    "LASR Labs (London AI Safety Research Labs)": ("LASR Labs", "program"),
    "Cambridge Boston Alignment Initiative (CBAI)": ("CBAI", "program"),
    "CBAI (Cambridge Boston Alignment Initiative)": ("CBAI", "program"),
    "Harvard AI Safety Team (HAIST)": ("Harvard AI Safety Team", "program"),
    "Harvard AI Safety Student Team (HAISST)": ("Harvard AI Safety Team", "program"),
    "Rutgers University WINLAB (Summer Internship Program)": ("Rutgers WINLAB", "program"),
    "Simons Institute for the Theory of Computing (UC Berkeley)": ("Simons Institute (Berkeley)", "program"),
    "Cosmos Institute / FIRE — AI x Truth-Seeking Grant": ("Cosmos Institute / FIRE", "program"),
    "OpenDP (Harvard)": ("OpenDP (Harvard)", "program"),
    # --- Harvard schools fold to the university (roles keep the school detail) ---
    "Harvard University": ("Harvard University", "university"),
    "Harvard Kennedy School": ("Harvard University", "university"),
    "Harvard Kennedy School (Belfer Center)": ("Harvard University", "university"),
    "Belfer Center for Science and International Affairs (Harvard)": ("Harvard University", "university"),
    "Harvard Business School": ("Harvard University", "university"),
    "Harvard John A. Paulson School of Engineering and Applied Sciences (SEAS)": ("Harvard University", "university"),
    # --- Microsoft fold (Research divisions are still the same employer) ---
    "Microsoft": ("Microsoft", "company"),
    "Microsoft (Research / Special Projects)": ("Microsoft", "company"),
    "Microsoft Research": ("Microsoft", "company"),
    "Microsoft Research (AI Frontiers)": ("Microsoft", "company"),
    # --- communities / platforms ---
    "Kaggle": ("Kaggle", "community"),
    "San Diego Machine Learning": ("San Diego Machine Learning", "community"),
    "San Diego Machine Learning (meetup)": ("San Diego Machine Learning", "community"),
    "Vesuvius Challenge (Kaggle Ink Detection, 1st place team)": ("Vesuvius Challenge team", "community"),
    "Mechanistic Interpretability Discord (mech interp community)": ("Mech Interp Discord", "community"),
    "EleutherAI": ("EleutherAI", "community"),
    # --- labs with verbose names ---
    "EPFL (Data Science Lab / dlab, Robert West)": ("EPFL Data Science Lab (dlab)", "lab"),
    "EPFL Data Science Lab (DLAB)": ("EPFL Data Science Lab (dlab)", "lab"),
    "MIT CSAIL": ("MIT CSAIL", "lab"),
    "MIT CSAIL — Computational Connectomics group": ("MIT CSAIL", "lab"),
    "Center for Human-Compatible AI (CHAI), UC Berkeley": ("CHAI (UC Berkeley)", "lab"),
    "NSF AI Institute for Artificial Intelligence and Fundamental Interactions (IAIFI)": ("IAIFI (NSF AI Institute)", "lab"),
    "Visual Inference Lab (Kriegeskorte Lab), Columbia Zuckerman Institute": ("Kriegeskorte Lab (Columbia)", "lab"),
    "OLAB (Oermann Lab), NYU Langone Health": ("Oermann Lab (NYU Langone)", "lab"),
    "Leuthardt Lab, Washington University Neurosurgery": ("Leuthardt Lab (WashU)", "lab"),
    "LUNAR Lab, Brown University (Ellie Pavlick)": ("LUNAR Lab (Brown)", "lab"),
    "David Krueger's lab (University of Cambridge)": ("Krueger Lab (Cambridge)", "lab"),
    "Dynamic Robotics Laboratory (Oregon State University)": ("Dynamic Robotics Lab (OSU)", "lab"),
    "New York University (ML2 group)": ("NYU ML2 group", "lab"),
    "Mila (Quebec Artificial Intelligence Institute)": ("Mila", "lab"),
    "Stanford Digital Economy Lab": ("Stanford Digital Economy Lab", "lab"),
    "UK AI Security Institute (AISI)": ("UK AI Security Institute", "lab"),
    # --- universities: shorter familiar labels, comma-variant folds ---
    "Massachusetts Institute of Technology": ("MIT", "university"),
    "Georgia Institute of Technology": ("Georgia Tech", "university"),
    "University of California, Berkeley": ("UC Berkeley", "university"),
    "University of California, San Diego": ("UC San Diego", "university"),
    "University of California San Diego": ("UC San Diego", "university"),
    "University of Pennsylvania": ("University of Pennsylvania", "university"),
    "ENS Paris-Saclay (École Normale Supérieure Paris-Saclay)": ("ENS Paris-Saclay", "university"),
    "University of Innsbruck (LFU Innsbruck)": ("University of Innsbruck", "university"),
    "Technische Universität Wien (TU Wien)": ("TU Wien", "university"),
    "Shahjalal University of Science and Technology (SUST)": ("Shahjalal University (SUST)", "university"),
    "ETH Zurich (Advanced Computing Laboratory)": ("ETH Zurich", "university"),
    "KTH Royal Institute of Technology": ("KTH", "university"),
    "Washington University in St. Louis School of Medicine": ("WashU School of Medicine", "university"),
    "University of Science and Technology of China (USTC)": ("USTC", "university"),
    # --- companies: trim legalese ---
    "D. E. Shaw & Co.": ("D. E. Shaw", "company"),
    "Anduril Industries": ("Anduril", "company"),
    "PricewaterhouseCoopers (Canada)": ("PwC Canada", "company"),
    "Leap Labs (Leap Laboratories)": ("Leap Labs", "company"),
    "Astera Institute (Simplex)": ("Astera Institute / Simplex", "company"),
    "Atana (later Operator.io)": ("Atana / Operator.io", "company"),
    "nference, Inc.": ("nference", "company"),
    "SMTM (Daughters Capital)": ("SMTM (Daughters Capital)", "company"),
}

# entries that aren't an organization at all — dropped from the graph (the person's sidebar
# detail still shows everything in affiliations.json; this only affects the drawn network)
DROP = {"Independent (mechanistic interpretability research)"}

# org containment, by canonical label: child -> the university it sits inside. Used to discount
# the projection — a pair sharing a lab AND its university shares one chapter of life, not two
# (e.g. all 10 NeuroData members also hold JHU degrees), so only the more specific org counts.
# Both nodes stay in the bipartite view. Both sides must exist as canonical orgs (asserted).
PARENT: dict[str, str] = {
    "NeuroData Lab (Johns Hopkins)": "Johns Hopkins University",
    "MINDS (Johns Hopkins)": "Johns Hopkins University",
    "Mao Lab (Johns Hopkins)": "Johns Hopkins University",
    "Saria Lab (Johns Hopkins)": "Johns Hopkins University",
    "JHU Applied Physics Laboratory": "Johns Hopkins University",
    "Bau Lab (Northeastern / NDIF)": "Northeastern University",
    "MIT CSAIL": "MIT",
    "IAIFI (NSF AI Institute)": "MIT",
    "CHAI (UC Berkeley)": "UC Berkeley",
    "Simons Institute (Berkeley)": "UC Berkeley",
    "LUNAR Lab (Brown)": "Brown University",
    "Kriegeskorte Lab (Columbia)": "Columbia University",
    "Krueger Lab (Cambridge)": "University of Cambridge",
    "Dynamic Robotics Lab (OSU)": "Oregon State University",
    "Leuthardt Lab (WashU)": "WashU School of Medicine",
    "Oermann Lab (NYU Langone)": "New York University",
    "NYU ML2 group": "New York University",
    "Stanford Digital Economy Lab": "Stanford University",
    "Harvard AI Safety Team": "Harvard University",
    "OpenDP (Harvard)": "Harvard University",
}


def slug(s: str) -> str:
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")


def normalize_years(specs: list[tuple[str, bool]]) -> str:
    """Fold one link's contributing (years, current) entries into one honest span.

    A trailing dash means "ongoing" only when some contributing entry is current=True —
    the sweep uses dash + current=False for "started X, end unknown" (emit just the start,
    never an open span that draws as still-there-today). Merged stints span from the
    earliest start; e.g. fred's three Harvard schools (2022/2024) -> "2022–", and a closed
    PhD stint plus a current research post -> open span from the PhD's start."""
    starts = [int(m.group(1)) for y, _ in specs if (m := re.match(r"^\s*(\d{4})", y or ""))]
    ends = [int(re.findall(r"\d{4}", y)[-1]) for y, _ in specs
            if y and re.findall(r"\d{4}", y) and not re.search(r"[–-]\s*$", y)]
    if not starts and not ends:
        return ""
    if not starts:
        return f"–{max(ends)}"
    start = min(starts)
    if any(cur is True for _, cur in specs):
        return f"{start}–"
    end = max(ends) if ends else None
    return f"{start}–{end}" if end and end > start else f"{start}"


def load_src() -> dict:
    """The hand-curated source records with the self-service overlay applied (in memory).

    affiliations.json itself is never machine-written; site edits/joins live in
    affiliation_overrides.json (regenerated nightly from the event log) and merge here."""
    src = json.loads(SRC.read_text())
    merged, warnings = apply_aff_overlay(src, load_aff_overrides(OVERRIDES), canon=CANON)
    for w in warnings:
        print(f"  overlay: {w}")
    # the registry decides WHO is on the maps; roster people with no chapters yet still ship
    merged, added = reconcile_membership(merged, ROSTER_PATH)
    for n in added:
        print(f"  registry: {n} has no recorded chapters yet — shipping an empty seat")
    return merged


PERSON_PAGE_MARKER = "<!-- generated by build_affiliations.py — do not edit -->"
PERSON_PAGES_DIR = REPO / "networks"


def write_person_pages(people: list[dict]) -> None:
    """Clean share URLs: /networks/<slug>/ -> tiny static page that titles the link preview and
    forwards to the person's seat. Generated dirs carry a marker; stale ones are removed (only
    marker-bearing dirs are ever touched). Unknown/future slugs fall back to the 404 router."""
    PERSON_PAGES_DIR.mkdir(exist_ok=True)
    want = {}
    for p in people:
        slug_id = p["id"].replace(" ", "-")
        target = f"/networks/affiliations/analyses/?p={p['id'].replace(' ', '%20')}#your-seat"
        want[slug_id] = (
            f"{PERSON_PAGE_MARKER}\n<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">\n"
            f"<title>{p['label']}\u2019s seat \u2014 the network</title>\n"
            f"<meta property=\"og:title\" content=\"{p['label']}\u2019s seat in the network\">\n"
            f"<meta property=\"og:description\" content=\"Ties, invitations, chapters, and outposts \u2014 "
            f"{p['label']}\u2019s personal read of the network.\">\n"
            f"<link rel=\"canonical\" href=\"https://alex-loftus.com{target}\">\n"
            f"<meta http-equiv=\"refresh\" content=\"0; url={target}\">\n"
            f"<script>location.replace(\"{target}\");</script>\n"
            f"</head><body><p><a href=\"{target}\">{p['label']}\u2019s seat \u2192</a></p></body></html>\n"
        )
    for d in PERSON_PAGES_DIR.iterdir():
        idx = d / "index.html"
        if d.is_dir() and idx.exists() and PERSON_PAGE_MARKER in idx.read_text():
            if d.name not in want:
                idx.unlink()
                d.rmdir()
    for slug_id, html in want.items():
        d = PERSON_PAGES_DIR / slug_id
        d.mkdir(exist_ok=True)
        (d / "index.html").write_text(html)


def main() -> None:
    people_src = load_src()
    graph = json.loads(GRAPH.read_text())
    map_nodes = {norm(n["id"]): n for n in graph["nodes"] if n.get("is_list")}
    map_nodes = {name: map_nodes[norm(name)] for name in people_src if norm(name) in map_nodes}
    missing = sorted(set(people_src) - set(map_nodes))
    assert not missing, f"people absent from coauthorship.json: {missing}"
    # everything shipped uses the map's canonical node id (e.g. "leo mckee-reid" -> "leo mckee reid")
    # so the affiliation data, the analyses shared file, and coauthorship.json all join on one key
    cid = {name: map_nodes[name]["id"] for name in people_src}
    assert len(set(cid.values())) == len(cid), "canonical id collision"

    # --- canonicalize entries; merge per-person duplicates created by the fold ---
    orgs: dict[str, dict] = {}                     # org id -> {label, type, members}
    links: list[dict] = []
    dropped: list[tuple[str, str]] = []
    for person in sorted(people_src, key=lambda n: cid[n]):
        per_org: dict[str, dict] = {}
        for e in people_src[person]["entries"]:
            if e["org"] in DROP:
                dropped.append((person, e["org"]))
                continue
            label, typ = CANON.get(e["org"], (e["org"], e["type"]))
            assert typ in TYPE_WEIGHT, (person, e["org"], typ)
            oid = slug(label)
            prev = orgs.setdefault(oid, {"label": label, "type": typ, "members": set()})
            assert (prev["label"], prev["type"]) == (label, typ), \
                f"slug collision or type conflict at {oid}: {(prev['label'], prev['type'])} vs {(label, typ)}"
            prev["members"].add(cid[person])
            if oid in per_org:                     # same canonical org twice (e.g. fred's 3 Harvard schools)
                m = per_org[oid]
                roles = [r for r in (m["role"], e.get("role", "")) if r]
                m["role"] = "; ".join(dict.fromkeys(roles))   # unique, order-preserving
                m["_years"].append((e.get("years", ""), e.get("current")))
            else:
                per_org[oid] = {
                    "person": cid[person], "org": oid,
                    "role": e.get("role", ""),
                    "_years": [(e.get("years", ""), e.get("current"))],
                    "source": e["source"],
                }
        for k in sorted(per_org):
            m = per_org[k]
            m["years"] = normalize_years(m.pop("_years"))
            m = {f: m[f] for f in ("person", "org", "role", "years", "source")}  # stable key order
            links.append(m)

    # --- person–person projection: type-weighted shared orgs, nesting-discounted ---
    parent_of: dict[str, str] = {}
    for child, parent in PARENT.items():
        child_id, parent_id = slug(child), slug(parent)
        assert child_id in orgs, f"PARENT child not in graph: {child}"
        assert parent_id in orgs, f"PARENT parent not in graph: {parent}"
        parent_of[child_id] = parent_id

    pair_shared: dict[tuple[str, str], list[str]] = defaultdict(list)
    for oid, o in orgs.items():
        members = sorted(o["members"])
        for i, a in enumerate(members):
            for b in members[i + 1:]:
                pair_shared[(a, b)].append(oid)
    projection = []
    for (a, b), shared in sorted(pair_shared.items()):
        # a shared lab implies its university — count only the more specific org
        implied = {parent_of[o] for o in shared if o in parent_of}
        kept = sorted(o for o in shared if o not in implied)
        projection.append({
            "a": a, "b": b,
            "weight": round(sum(TYPE_WEIGHT[orgs[o]["type"]] for o in kept), 1),
            "shared": kept,
        })

    people = [
        {
            "id": cid[name],
            "label": map_nodes[name]["label"],
            "initials": map_nodes[name]["initials"],
            "community": map_nodes[name]["community"],
            "city": people_src[name].get("city", ""),
        }
        for name in sorted(people_src, key=lambda n: cid[n])
    ]
    org_list = [
        {"id": oid, "label": o["label"], "type": o["type"], "n_members": len(o["members"])}
        for oid, o in sorted(orgs.items())
    ]

    # --- fail-fast invariants ---
    assert len(people) == len(people_src) >= 48, (len(people), len(people_src))
    oid_set, pid_set = {o["id"] for o in org_list}, {p["id"] for p in people}
    assert all(l["org"] in oid_set and l["person"] in pid_set for l in links)
    member_counts = defaultdict(int)
    for l in links:
        member_counts[l["org"]] += 1
    assert all(member_counts[o["id"]] == o["n_members"] for o in org_list)
    assert all(p["a"] < p["b"] for p in projection)

    out = {
        "people": people, "orgs": org_list, "links": links, "projection": projection,
        "communities": graph["communities"],
        # parent ships so the map page can replicate the nesting discount client-side (live preview)
        "meta": {"type_weights": TYPE_WEIGHT, "parent": parent_of},
    }
    OUT.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")

    # --- shared lookup for the affiliation analyses page (same schema as analyses/shared.json,
    # population = the 48 people only, degree = projection degree, links feed the minimap) ---
    deg = defaultdict(int)
    for p in projection:
        deg[p["a"]] += 1
        deg[p["b"]] += 1
    SHARED_OUT.parent.mkdir(parents=True, exist_ok=True)
    SHARED_OUT.write_text(json.dumps({
        "nodes": {p["id"]: {"label": p["label"], "community": p["community"],
                            "is_list": True, "degree": deg[p["id"]]} for p in people},
        "links": [[p["a"], p["b"], p["weight"]] for p in projection],
        "communities": graph["communities"],
    }, separators=(",", ":"), ensure_ascii=False) + "\n")

    shared = sorted((o for o in org_list if o["n_members"] >= 2), key=lambda o: -o["n_members"])
    write_person_pages(people)

    rel = lambda p: p.relative_to(REPO) if p.is_relative_to(REPO) else p
    print(f"wrote {rel(OUT)} + {rel(SHARED_OUT)}: {len(people)} people, "
          f"{len(org_list)} orgs ({len(shared)} shared), {len(links)} links, {len(projection)} projected pairs")
    for o in shared[:12]:
        print(f"  {o['n_members']:2d}  [{o['type']}] {o['label']}")
    for p, o in dropped:
        print(f"  dropped non-org entry: {p}: {o}")


if __name__ == "__main__":
    main()
