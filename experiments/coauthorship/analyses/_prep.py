# /// script
# requires-python = ">=3.10"
# dependencies = ["scikit-learn", "httpx"]
# ///
"""Shared data prep for /coauthorship/analyses/ — run ONCE before panel agents.

Re-keys per-paper records from the local raw API cache (gitignored) using the SAME identity
logic as build_graph.py, then emits the derived inputs every panel shares:

    _derived/papers.json    every distinct paper by a listed person (year, members, sources)
    _derived/yearly.json    per-year + cumulative weighted edge stacks on a fixed vertex order
    _derived/layers.json    s2-only / openalex-only edge slices (same vertex order)
    _derived/tfidf.json     title TF-IDF vectors per list member
    ../../../public/assets/data/analyses/shared.json   small browser lookup (labels/communities/degree)

Cache-first: reads the same v2 cache pages build_graph.py writes; a missing page is fetched
from the live API (same fields, same backoff) and cached, so the first run populates and
every later run is offline-deterministic.

CROSS-CHECK GATE: re-derives every shipped link's weight from its own paper records and
requires >=95% exact agreement with assets/data/coauthorship.json. If this fails, the
re-keying drifted from build_graph.py — fix _prep, don't ship.

Run: cd experiments/coauthorship/analyses && uv run _prep.py
"""
import html
import json
import re
import unicodedata
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
COAUTH = HERE.parent                      # experiments/coauthorship
REPO = HERE.parents[2]
RAW = COAUTH / "raw"
SEEDS = COAUTH / "seeds.json"
DERIVED = HERE / "_derived"
DERIVED.mkdir(exist_ok=True)
SHARED_OUT = REPO / "public" / "assets" / "data" / "analyses" / "shared.json"
GRAPH = json.loads((REPO / "public" / "assets" / "data" / "coauthorship.json").read_text())

# ---- vendored from build_graph.py (lines 49-136) — KEEP IN SYNC -----------------------------
MAX_AUTHORS = 25
COMMON_STOP = {
    "xin chen", "y liu", "jamie simon", "wei wang", "jun zhang", "wei li", "yang liu",
    "li wang", "jing wang", "yan zhang", "lei zhang", "jie chen", "wei zhang", "xin li",
}
NAME_ALIAS = {"c priebe": "carey e priebe"}
COLLISION_STOP = {"richard guo", "j smith"}
EDGE_DROP = {tuple(sorted(("jack merullo", "jannik brinkmann")))}


def clean_title(t: str) -> str:
    return html.unescape(re.sub(r"<[^>]+>", "", t or "")).strip()


def clean_display(s: str) -> str:
    if "," in s:
        parts = [p.strip() for p in s.split(",", 1)]
        if len(parts) == 2 and parts[1]:
            s = f"{parts[1]} {parts[0]}"
    return s


def norm(s: str) -> str:
    s = clean_display(s)
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower().replace("-", " ")
    return " ".join(re.sub(r"[^\w\s]", " ", s).split())


def _main_title(title: str) -> str:
    head = norm(title.split(":", 1)[0])
    return head if len(head.split()) >= 3 else ""


def same_paper(t1: str, t2: str) -> bool:
    n1, n2 = norm(t1), norm(t2)
    if not n1 or not n2:
        return False
    if n1 == n2:
        return True
    s1, s2 = set(n1.split()), set(n2.split())
    if s1 <= s2 or s2 <= s1:
        return True
    m1, m2 = _main_title(t1), _main_title(t2)
    return bool(m1) and m1 == m2


def distinct_title_list(titles: list[str]) -> list[str]:
    clusters: list[str] = []
    for t in titles:
        if not any(same_paper(t, rep) for rep in clusters):
            clusters.append(t)
    return clusters


def distinct_papers(titles: list[str]) -> int:
    return len(distinct_title_list(titles))


def fractional_weight(records: list[tuple[str, int]]) -> float:
    """Mirror of build_graph.fractional_weight — KEEP IN SYNC. Each distinct paper adds
    1/n_authors to its pairs (fractional counting, suggested by Stella Biderman); version
    variants collapse first, each cluster contributing 1/min(n) across its variants."""
    clusters: list[list[tuple[str, int]]] = []
    for t, n in records:
        for c in clusters:
            if same_paper(t, c[0][0]):
                c.append((t, n))
                break
        else:
            clusters.append([(t, n)])
    return sum(1.0 / max(1, min(n for _, n in c)) for c in clusters)


def oa_bare(oa_url) -> str:
    return (oa_url or "").rstrip("/").split("/")[-1]
# ---- end vendored ----------------------------------------------------------------------------


# ---- cached fetch, mirroring build_graph.cached_get (network fallback on cache miss) --------
import os                                                                       # noqa: E402
import time                                                                     # noqa: E402
import httpx                                                                    # noqa: E402

S2_BASE = "https://api.semanticscholar.org/graph/v1"
OA_BASE = "https://api.openalex.org"
MAILTO = "alexloftus2004@gmail.com"
_HEADERS = {"User-Agent": f"coauthorship-build ({MAILTO})"}
if os.environ.get("S2_API_KEY"):
    _HEADERS["x-api-key"] = os.environ["S2_API_KEY"]
_s2 = httpx.Client(timeout=60, headers=_HEADERS)
_oa = httpx.Client(timeout=60, headers={"User-Agent": f"coauthorship-build ({MAILTO})"})


def cached_get(client: httpx.Client, base: str, path: str, cache_key: str, **params) -> dict:
    f = RAW / f"{cache_key}.json"
    if f.exists():
        return json.loads(f.read_text())
    for attempt in range(10):
        try:
            r = client.get(f"{base}/{path}", params=params)
        except httpx.HTTPError:
            time.sleep(1.5 * (attempt + 1))
            continue
        if r.status_code == 200:
            data = r.json()
            f.write_text(json.dumps(data))
            return data
        if r.status_code == 429:
            time.sleep(2.5 * (attempt + 1))
            continue
        r.raise_for_status()
    raise RuntimeError(f"giving up on {base}/{path} after repeated errors")


def works_s2(author_id: str) -> list[dict]:
    """Mirror of build_graph.fetch_works (v2 cache keys; fetches+caches on miss)."""
    out, offset = [], 0
    while True:
        data = cached_get(_s2, S2_BASE, f"author/{author_id}/papers",
                          f"coauth_papers_v2_{author_id}_{offset}",
                          fields="title,year,citationCount,externalIds,authors.authorId,authors.name",
                          limit=500, offset=offset)
        for p in data.get("data", []):
            out.append({
                "id": p["paperId"],
                "doi": ((p.get("externalIds") or {}).get("DOI") or "").lower() or None,
                "title": clean_title(p.get("title")),
                "publication_year": p.get("year"),
                "citations": p.get("citationCount") or 0,
                "authorships": [{"author": {"id": a.get("authorId"), "display_name": a.get("name") or ""}}
                                for a in (p.get("authors") or []) if a.get("name")],
            })
        if not data.get("next"):
            break
        offset = data["next"]
    return out


def works_oa(author_id: str) -> list[dict]:
    """Mirror of build_graph.fetch_works_openalex (v2 cache keys; fetches+caches on miss)."""
    out, cursor, page = [], "*", 0
    while cursor:
        data = cached_get(_oa, OA_BASE, "works", f"oa_works_v2_{author_id}_{page}",
                          mailto=MAILTO, filter=f"author.id:{author_id}", per_page=200, cursor=cursor,
                          select="id,doi,title,publication_year,cited_by_count,authorships")
        for w in data.get("results", []):
            auths = []
            for a in w.get("authorships", []):
                nm = (a.get("author") or {}).get("display_name") or a.get("raw_author_name") or ""
                if nm:
                    auths.append({"author": {"id": oa_bare((a.get("author") or {}).get("id")), "display_name": nm}})
            out.append({
                "id": oa_bare(w["id"]),
                "doi": (w.get("doi") or "").replace("https://doi.org/", "").lower() or None,
                "title": clean_title(w.get("title")),
                "publication_year": w.get("publication_year"),
                "citations": w.get("cited_by_count") or 0,
                "authorships": auths,
            })
        cursor = (data.get("meta") or {}).get("next_cursor")
        page += 1
        if not data.get("results"):
            break
    return out


# ---- 1. seeds + works (same trust gate as build_graph section 1-2) ---------------------------
seeds = json.loads(SEEDS.read_text())
seed_by_norm = {norm(e["name"]): e for e in seeds}
person_ids, person_oa_ids = {}, {}
for e in seeds:
    name = norm(e["name"])
    person_ids[name] = [str(i) for i in ([e["s2_id"]] + e.get("merge_ids", [])) if i]
    cc = e.get("crosscheck") or {}
    trust_oa = bool(e.get("oa_verified") or cc.get("agree")) and not e.get("oa_reject")
    person_oa_ids[name] = [oa_bare(i) for i in ([e.get("oa_id")] + e.get("oa_merge_ids", [])) if i] if trust_oa else []
resolved = [n for n in person_ids if person_ids[n]]

person_works, alias_to_canon = {}, {}
for q in resolved:
    works = []
    for aid in person_ids[q]:
        works += [dict(w, source="s2") for w in works_s2(aid)]
    for aid in person_oa_ids.get(q, []):
        works += [dict(w, source="openalex") for w in works_oa(aid)]
    excl = [norm(s) for s in (seed_by_norm.get(q, {}).get("exclude_titles") or []) if s]
    if excl:
        works = [w for w in works if not (w.get("title") and any(x in norm(w["title"]) for x in excl))]
    person_works[q] = works
    alias_to_canon[q] = q
    s2set, oaset = set(person_ids[q]), set(person_oa_ids.get(q, []))
    for w in works:
        idset = s2set if w["source"] == "s2" else oaset
        for a in w["authorships"]:
            if str(a["author"].get("id")) in idset:
                alias_to_canon[norm(a["author"]["display_name"])] = q
print(f"works fetched from cache for {len(resolved)} people")


def canon(nm: str) -> str:
    nm = NAME_ALIAS.get(nm, nm)
    return alias_to_canon.get(nm, nm)


def paper_key(w: dict) -> str:
    if w.get("doi"):
        return "doi:" + w["doi"]
    if w.get("title"):
        return "t:" + norm(w["title"])
    return f'{w["source"]}:{w["id"]}'


# ---- 2. cross-source paper dedup (build_graph section 3) + year/citations --------------------
papers: dict[str, dict] = {}
for q in resolved:
    for w in person_works[q]:
        rec = papers.setdefault(paper_key(w), {"authors": {}, "sources": set(),
                                               "title": w.get("title", ""), "year": None, "citations": 0})
        rec["sources"].add(w["source"])
        if not rec["title"] and w.get("title"):
            rec["title"] = w["title"]
        if rec["year"] is None and w.get("publication_year"):
            rec["year"] = w["publication_year"]
        rec["citations"] = max(rec["citations"], w.get("citations") or 0)
        for a in w["authorships"]:
            nm = norm(a["author"]["display_name"])
            if nm:
                rec["authors"].setdefault(nm, a["author"]["display_name"])

shipped_ids = {n["id"] for n in GRAPH["nodes"]}
list_nodes = {n["id"] for n in GRAPH["nodes"] if n.get("is_list")}
vertex_order = sorted(n["id"] for n in GRAPH["nodes"] if not n.get("path_only"))
vidx = {v: i for i, v in enumerate(vertex_order)}

paper_rows = []
for key, rec in sorted(papers.items()):
    keys = sorted({canon(nm) for nm in rec["authors"]})
    members = [k for k in keys if k in shipped_ids]
    paper_rows.append({
        "key": key, "title": rec["title"], "year": rec["year"], "citations": rec["citations"],
        "sources": sorted(rec["sources"]), "members": members,
        "n_authors_total": len(rec["authors"]), "big": len(rec["authors"]) > MAX_AUTHORS,
    })
print(f"papers: {len(paper_rows)} distinct records, "
      f"{sum(1 for p in paper_rows if p['year'])} with a year, "
      f"{sum(1 for p in paper_rows if len(p['members']) >= 2)} with >=2 graph members")


def pairs_of(p):
    """Same edge rule as build_graph: big papers only contribute list<->list ties."""
    mem = p["members"] if not p["big"] else [m for m in p["members"] if m in list_nodes]
    out = []
    for i in range(len(mem)):
        for j in range(i + 1, len(mem)):
            e = tuple(sorted((mem[i], mem[j])))
            if e not in EDGE_DROP:
                out.append(e)
    return out


# ---- 3. cross-check gate: re-derive shipped link weights (float tolerance) -------------------
pair_recs = defaultdict(list)
for p in paper_rows:
    for e in pairs_of(p):
        pair_recs[e].append((p["title"], p["n_authors_total"]))
derived_w = {e: round(fractional_weight(recs), 4) for e, recs in pair_recs.items()}

links = GRAPH["links"]
TOL = 0.005   # build rounds to 4 decimals; anything past rounding noise is real drift
match = sum(1 for l in links
            if abs(derived_w.get(tuple(sorted((l["source"], l["target"]))), -9) - l["weight"]) <= TOL)
agree = match / len(links)
print(f"CROSS-CHECK: {match}/{len(links)} shipped link weights reproduced (±{TOL}) ({agree:.1%})")
for l in links:
    e = tuple(sorted((l["source"], l["target"])))
    if abs(derived_w.get(e, -9) - l["weight"]) > TOL:
        print(f"  mismatch {e}: shipped={l['weight']} derived={derived_w.get(e)}")
assert agree >= 0.95, f"re-keying drifted from build_graph ({agree:.1%} < 95%) — fix _prep.py"

# ---- 4. yearly + cumulative edge stacks ------------------------------------------------------
years = sorted({p["year"] for p in paper_rows if p["year"] and any(
    a in vidx and b in vidx for a, b in pairs_of(p))})
per_year_titles = defaultdict(lambda: defaultdict(list))   # year -> pair -> (title, n_authors)
for p in paper_rows:
    if not p["year"]:
        continue
    for e in pairs_of(p):
        if e[0] in vidx and e[1] in vidx:
            per_year_titles[p["year"]][e].append((p["title"], p["n_authors_total"]))

per_year, cumulative = {}, {}
cum_titles = defaultdict(list)
for y in years:
    per_year[y] = [[vidx[a], vidx[b], round(fractional_weight(ts), 4)]
                   for (a, b), ts in sorted(per_year_titles[y].items())]
    for e, ts in per_year_titles[y].items():
        cum_titles[e].extend(ts)
    cumulative[y] = [[vidx[a], vidx[b], round(fractional_weight(ts), 4)]
                     for e, ts in sorted(cum_titles.items())
                     for a, b in [e]]
print(f"yearly stacks: {len(years)} years ({years[0]}..{years[-1]}), "
      f"{sum(len(v) for v in per_year.values())} year-edges, final cumulative {len(cumulative[years[-1]])} edges")

# ---- 5. source layers ------------------------------------------------------------------------
layer_titles = {"s2": defaultdict(list), "oa": defaultdict(list)}
for p in paper_rows:
    tgt = []
    if "s2" in p["sources"]:
        tgt.append("s2")
    if "openalex" in p["sources"]:
        tgt.append("oa")
    for e in pairs_of(p):
        if e[0] in vidx and e[1] in vidx:
            for t in tgt:
                layer_titles[t][e].append((p["title"], p["n_authors_total"]))
layers = {t: [[vidx[a], vidx[b], round(fractional_weight(ts), 4)] for (a, b), ts in sorted(d.items())]
          for t, d in layer_titles.items()}
print(f"layers: s2={len(layers['s2'])} edges, oa={len(layers['oa'])} edges")

# ---- 6. TF-IDF title vectors per list member -------------------------------------------------
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402

docs, ids = [], []
for m in sorted(list_nodes):
    titles = distinct_title_list([p["title"] for p in paper_rows if m in p["members"] and p["title"]])
    if titles:
        ids.append(m)
        docs.append(" ".join(titles))
vec = TfidfVectorizer(stop_words="english", min_df=2, max_features=500)
X = vec.fit_transform(docs)
rows = [[[int(j), round(float(X[i, j]), 4)] for j in X[i].nonzero()[1]] for i in range(X.shape[0])]
print(f"tfidf: {len(ids)} members x {len(vec.get_feature_names_out())} terms")

# ---- 7. write outputs ------------------------------------------------------------------------
(DERIVED / "papers.json").write_text(json.dumps(paper_rows, separators=(",", ":")))
(DERIVED / "yearly.json").write_text(json.dumps(
    {"vertex_order": vertex_order, "years": years,
     "per_year": {str(y): per_year[y] for y in years},
     "cumulative": {str(y): cumulative[y] for y in years}}, separators=(",", ":")))
(DERIVED / "layers.json").write_text(json.dumps(
    {"vertex_order": vertex_order, "s2": layers["s2"], "oa": layers["oa"]}, separators=(",", ":")))
(DERIVED / "tfidf.json").write_text(json.dumps(
    {"ids": ids, "vocab": vec.get_feature_names_out().tolist(), "rows": rows}, separators=(",", ":")))

SHARED_OUT.parent.mkdir(parents=True, exist_ok=True)
SHARED_OUT.write_text(json.dumps({
    "nodes": {n["id"]: {"label": n["label"], "community": n.get("community", -1),
                        "is_list": bool(n.get("is_list")), "degree": n.get("degree", 0)}
              for n in GRAPH["nodes"]},
    "communities": GRAPH["communities"],
    "years": {"min": years[0], "max": years[-1]},
}, separators=(",", ":")))

for f in ["papers", "yearly", "layers", "tfidf"]:
    kb = (DERIVED / f"{f}.json").stat().st_size / 1024
    print(f"  _derived/{f}.json  {kb:.0f} KB")
print(f"  shared.json  {SHARED_OUT.stat().st_size / 1024:.0f} KB")
print("PREP OK")
