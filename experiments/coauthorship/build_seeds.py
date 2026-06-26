# /// script
# requires-python = ">=3.10"
# dependencies = ["httpx"]
# ///
"""Build a hand-verifiable seed file mapping each listed researcher to BOTH a Semantic Scholar and
an OpenAlex author, cross-referencing the two so wrong-profile picks get caught.

Both indices fragment one author across several profiles and occasionally over-merge a homonym
(OpenAlex especially). Rather than guess at view time, this resolves each name to its *dominant*
profile in each index ONCE, writes everything you need to eyeball-check it (affiliation, ORCID,
paper count, top-cited titles) plus the runner-up profiles inline, and a `crosscheck` block that
compares the two picks (shared ORCID, overlapping top titles). You verify/correct `seeds.json` by
hand, then `build_graph.py` unions papers from both pinned IDs — no fuzzy search, nothing guessed.

    cd experiments/coauthorship && uv run build_seeds.py

Output: seeds.json (one entry per name). Re-running PRESERVES an entry's existing hand-verified S2
side (s2_id, merge_ids, verified, …) and only (re)computes the OpenAlex side + crosscheck — so it
augments rather than clobbering your S2 verification. Raw API responses cached under raw/ so re-runs
are instant and offline. Set S2_API_KEY for higher S2 rate limits (OpenAlex uses the polite pool).

How to verify each entry:
  - `affiliation` / `homepage` / `orcid` / `sample_titles` match the person you mean? click `url`.
  - `crosscheck.agree == false` (no shared ORCID and no shared top title) → S2 and OpenAlex may have
    landed on different people; scrutinise these FIRST. `crosscheck.orcid_conflict == true` (both have
    an ORCID and they differ) is the strongest "one of these is wrong" signal.
  - if WRONG: replace `s2_id`/`oa_id` (+ url) — often the right profile is already in `alternatives`
    / `oa_alternatives`, else paste the id from their profile page. To fold a fragmented secondary
    profile in, add its id to `merge_ids` (S2) / `oa_merge_ids` (OpenAlex).
  - `confidence: "low"` entries are also worth scrutinising (homonyms / no clear S2 winner).
"""
import json
import os
import re
import time
import unicodedata
from pathlib import Path

import httpx

BASE = "https://api.semanticscholar.org/graph/v1"
OA_BASE = "https://api.openalex.org"
MAILTO = "alexloftus2004@gmail.com"          # OpenAlex polite pool
HERE = Path(__file__).resolve().parent
OUT = HERE / "seeds.json"
RAW = HERE / "raw"
RAW.mkdir(exist_ok=True)

# ---- the roster lives in roster.json: `core` is Alex's hand list; `self_joined` is
# machine-owned by merge_affiliations.py (people who added themselves via the site) ----
_ROSTER = json.loads((HERE / "roster.json").read_text())
NAMES = _ROSTER["core"] + _ROSTER["self_joined"]
# spelling corrections used only for the search query
SEARCH = {
    "arman ohmid": "arman omid", "cat khor": "catherine khor",
    "madi kusmonov": "madi kusmanov", "sarah schwettman": "sarah schwettmann",
}


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower().replace("-", " ")
    return " ".join(re.sub(r"[^\w\s]", " ", s).split())


def name_matches(query: str, candidate: str) -> bool:
    """First and last token of the query both appear in the candidate's name."""
    q, c = norm(query).split(), set(norm(candidate).split())
    return bool(q) and q[0] in c and q[-1] in c


HEADERS = {"User-Agent": "coauthorship-seeds (alexloftus2004@gmail.com)"}
if os.environ.get("S2_API_KEY"):
    HEADERS["x-api-key"] = os.environ["S2_API_KEY"]
client = httpx.Client(timeout=60, headers=HEADERS)


def get(path: str, cache_key: str, **params) -> dict:
    """GET with disk cache + backoff on 429. cache_key names the raw/<key>.json file."""
    cache = RAW / f"{cache_key}.json"
    if cache.exists():
        return json.loads(cache.read_text())
    for attempt in range(8):
        r = client.get(f"{BASE}/{path}", params=params)
        if r.status_code == 200:
            data = r.json()
            cache.write_text(json.dumps(data))
            return data
        if r.status_code == 429:
            wait = 2.0 * (attempt + 1)
            print(f"    429, backing off {wait:.0f}s (attempt {attempt + 1}/8)")
            time.sleep(wait)
            continue
        r.raise_for_status()
    raise RuntimeError(f"giving up on {path} after repeated 429s")


def search_candidates(query: str) -> list[dict]:
    fields = "name,affiliations,homepage,paperCount,hIndex,externalIds,url"
    data = get("author/search", f"search_{norm(query).replace(' ', '_')}",
               query=query, fields=fields, limit=15)
    return [a for a in data.get("data", []) if name_matches(query, a.get("name", ""))]


def top_titles(author_id: str, n: int = 5) -> list[str]:
    """The author's most-cited paper titles — the strongest signal for 'is this the right person?'."""
    data = get(f"author/{author_id}/papers", f"papers_{author_id}",
               fields="title,year,citationCount", limit=100)
    papers = [p for p in data.get("data", []) if p.get("title")]
    papers.sort(key=lambda p: p.get("citationCount") or 0, reverse=True)
    return [f"{p['title']} ({p.get('year') or '?'})" for p in papers[:n]]


def slim(a: dict) -> dict:
    return {
        "s2_id": a["authorId"],
        "url": a.get("url") or f"https://www.semanticscholar.org/author/{a['authorId']}",
        "affiliation": "; ".join(a.get("affiliations") or []),
        "n_papers": a.get("paperCount"),
        "h_index": a.get("hIndex"),
    }


# ---- OpenAlex side (cross-reference) ----------------------------------------
oa_client = httpx.Client(timeout=60, headers={"User-Agent": f"coauthorship-seeds ({MAILTO})"})


def oa_get(path: str, cache_key: str, **params) -> dict:
    cache = RAW / f"{cache_key}.json"
    if cache.exists():
        return json.loads(cache.read_text())
    params.setdefault("mailto", MAILTO)
    for attempt in range(8):
        try:
            r = oa_client.get(f"{OA_BASE}/{path}", params=params)
        except httpx.HTTPError:
            time.sleep(1.5 * (attempt + 1))
            continue
        if r.status_code == 200:
            data = r.json()
            cache.write_text(json.dumps(data))
            return data
        if r.status_code == 429:
            time.sleep(2.0 * (attempt + 1))
            continue
        r.raise_for_status()
    raise RuntimeError(f"giving up on OpenAlex {path}")


def oa_bare(oa_url: str) -> str:
    return (oa_url or "").rstrip("/").split("/")[-1]


def oa_search(query: str) -> list[dict]:
    data = oa_get("authors", f"oa_search_{norm(query).replace(' ', '_')}", search=query, per_page=15,
                  select="id,display_name,orcid,works_count,cited_by_count,last_known_institutions")
    return [a for a in data.get("results", []) if name_matches(query, a.get("display_name", ""))]


def oa_top_titles(oa_id: str, n: int = 5) -> list[str]:
    data = oa_get("works", f"oa_papers_{oa_id}", filter=f"author.id:{oa_id}", per_page=100,
                  sort="cited_by_count:desc", select="title,publication_year")
    titles = [w for w in data.get("results", []) if w.get("title")]
    return [f"{w['title']} ({w.get('publication_year') or '?'})" for w in titles[:n]]


def oa_slim(a: dict) -> dict:
    inst = a.get("last_known_institutions") or []
    return {
        "oa_id": oa_bare(a["id"]),
        "url": a["id"],
        "affiliation": inst[0]["display_name"] if inst else "",
        "n_works": a.get("works_count"),
        "cited_by": a.get("cited_by_count"),
        "orcid": oa_bare(a.get("orcid")) or None,
    }


def _oa_entry(prof: dict, oa_merge_ids: list[str], oa_alternatives: list[dict]) -> dict:
    """Assemble a seed's OpenAlex block from an author profile dict (shared by auto-pick + pinned)."""
    inst = prof.get("last_known_institutions") or []
    oa_id = oa_bare(prof["id"])
    return {
        "oa_id": oa_id,
        "oa_url": prof["id"],
        "oa_affiliation": inst[0]["display_name"] if inst else "",
        "oa_n_works": prof.get("works_count"),
        "oa_orcid": oa_bare(prof.get("orcid")) or None,
        "oa_titles": oa_top_titles(oa_id),
        "oa_merge_ids": oa_merge_ids or [],
        "oa_alternatives": oa_alternatives,
    }


def build_oa(query: str) -> dict:
    """Dominant OpenAlex profile for a name (by works_count), plus runner-ups for quick swap."""
    matches = oa_search(query)
    matches.sort(key=lambda a: a.get("works_count") or 0, reverse=True)
    if not matches:
        print(f"  {query:22s} OpenAlex: NOT FOUND — fill oa_id by hand")
        return {"oa_id": None, "oa_url": None, "oa_affiliation": "", "oa_n_works": None,
                "oa_orcid": None, "oa_titles": [], "oa_merge_ids": [], "oa_alternatives": []}
    # oa_merge_ids: add other OpenAlex ids here to fold extra profiles into this node
    return _oa_entry(matches[0], [], [oa_slim(a) for a in matches[1:6]])


def build_oa_pinned(oa_id: str, oa_merge_ids: list[str]) -> dict:
    """Re-fetch a HAND-PINNED OpenAlex profile (oa_verified) so titles/crosscheck stay fresh on
    re-run without the auto-picker clobbering the verified id (runner-ups no longer needed)."""
    return _oa_entry(oa_get(f"authors/{oa_id}", f"oa_author_{oa_id}"), oa_merge_ids, [])


def _title_keys(titles: list[str]) -> set[str]:
    """Normalize "Title (2024)" -> a comparable key, dropping the trailing (year)/(?)."""
    out = set()
    for t in titles:
        out.add(norm(re.sub(r"\s*\((?:\d{4}|\?)\)\s*$", "", t)))
    return {t for t in out if t}


def crosscheck(s2_orcid, oa_orcid, s2_titles, oa_titles) -> dict:
    """Compare the S2 and OpenAlex picks. ORCID is the gold key; shared top titles corroborate it."""
    s2t, oat = _title_keys(s2_titles or []), _title_keys(oa_titles or [])
    overlap = len(s2t & oat)
    denom = min(len(s2t), len(oat)) or 1
    s2o = (s2_orcid or "").lower() or None
    oao = (oa_orcid or "").lower() or None
    orcid_match = bool(s2o and oao and s2o == oao)
    orcid_conflict = bool(s2o and oao and s2o != oao)
    # agree if the ORCIDs match, or (absent ORCID) they share at least one top-cited paper
    agree = orcid_match or (not orcid_conflict and overlap >= 1)
    return {
        "orcid_s2": s2_orcid, "orcid_oa": oa_orcid,
        "orcid_match": orcid_match, "orcid_conflict": orcid_conflict,
        "title_overlap_n": overlap, "title_overlap_frac": round(overlap / denom, 2),
        "agree": bool(agree),
    }


def confidence(matches: list[dict]) -> str:
    if len(matches) == 1:
        return "high"
    top, second = (matches[0].get("paperCount") or 0), (matches[1].get("paperCount") or 0)
    return "high" if top >= 3 * max(second, 1) else "low"


def build_entry(query: str) -> dict:
    matches = search_candidates(query)
    matches.sort(key=lambda a: a.get("paperCount") or 0, reverse=True)
    if not matches:
        print(f"  {query:22s} NOT FOUND — fill s2_id by hand")
        return {"name": query, "s2_id": None, "url": None, "affiliation": "",
                "homepage": None, "orcid": None, "n_papers": None, "h_index": None,
                "confidence": "none", "verified": False, "sample_titles": [],
                "merge_ids": [], "alternatives": []}
    primary = matches[0]
    conf = confidence(matches)
    titles = top_titles(primary["authorId"])
    print(f"  {query:22s} -> {primary['authorId']:>11}  {(primary.get('affiliations') or [''])[0][:30]:30s}"
          f"  papers={primary.get('paperCount')}  [{conf}]")
    return {
        "name": query,
        "s2_id": primary["authorId"],
        "url": primary.get("url"),
        "affiliation": "; ".join(primary.get("affiliations") or []),
        "homepage": primary.get("homepage"),
        "orcid": (primary.get("externalIds") or {}).get("ORCID"),
        "n_papers": primary.get("paperCount"),
        "h_index": primary.get("hIndex"),
        "confidence": conf,            # "low" => scrutinise this one first
        "verified": False,             # flip to true once you've eyeballed it
        "sample_titles": titles,       # this person's top-cited papers
        "merge_ids": [],               # add other authorIds here to fold extra profiles into this node
        "alternatives": [slim(a) for a in matches[1:6]],  # other same-name profiles, for quick swap
    }


def main() -> None:
    print(f"Resolving {len(NAMES)} names against Semantic Scholar + OpenAlex"
          f"{' (S2 API key)' if 'x-api-key' in HEADERS else ' (S2 shared pool — may be slow)'}...\n")
    existing = {e["name"]: e for e in json.loads(OUT.read_text())} if OUT.exists() else {}
    entries = []
    for q in NAMES:
        query = SEARCH.get(q, q)
        prev = existing.get(q)
        # preserve an already hand-verified S2 side; only fresh-resolve S2 for brand-new names.
        # `verified` with s2_id=None is a hand-checked "no academic profile" — keep it, don't
        # let the auto-picker re-introduce a same-named stranger (e.g. Karl Kaiser the political
        # scientist, Maya Deen the Ghana researcher) on the next re-run.
        e = dict(prev) if (prev and (prev.get("s2_id") or prev.get("verified"))) else build_entry(query)
        e["name"] = q                       # canonical roster label, even when the search query differed
        oa_pin = prev.get("oa_id") if (prev and prev.get("oa_verified") and prev.get("oa_id")) else None
        # `oa_verified` with oa_id=None is the OA-side twin of the rule above: a hand-checked
        # "no OpenAlex profile" (the dominant pick was a homonym, e.g. daniel brown the
        # geographer) — keep it empty, don't re-auto-pick.
        oa_none = bool(prev and prev.get("oa_verified") and not prev.get("oa_id"))
        for k in [k for k in e if k.startswith("oa_") or k == "crosscheck"]:
            del e[k]                         # drop the stale OpenAlex/crosscheck block before re-adding
        # preserve a hand-pinned OpenAlex id (oa_verified); else auto-pick the dominant profile
        if oa_none:
            oa = {"oa_id": None, "oa_url": None, "oa_affiliation": "", "oa_n_works": None,
                  "oa_orcid": None, "oa_titles": [], "oa_merge_ids": [], "oa_alternatives": []}
        else:
            oa = build_oa_pinned(oa_pin, prev.get("oa_merge_ids")) if oa_pin else build_oa(query)
        if oa_pin or oa_none:
            e["oa_verified"] = True
        e.update(oa)
        e["crosscheck"] = crosscheck(e.get("orcid"), oa["oa_orcid"], e.get("sample_titles"), oa["oa_titles"])
        entries.append(e)

    low = [e["name"] for e in entries if e.get("confidence") == "low"]
    missing_s2 = [e["name"] for e in entries if e.get("s2_id") is None]
    missing_oa = [e["name"] for e in entries if e.get("oa_id") is None]
    disagree = [e["name"] for e in entries if e.get("s2_id") and e.get("oa_id")
                and not e["crosscheck"]["agree"]]
    conflict = [e["name"] for e in entries if e["crosscheck"]["orcid_conflict"]]
    OUT.write_text(json.dumps(entries, indent=2, ensure_ascii=False))
    print(f"\nwrote {OUT}  ({len(entries)} entries)")
    print(f"  ⚠️  S2/OpenAlex DISAGREE — check first ({len(disagree)}): {', '.join(disagree) or '—'}")
    print(f"  ⚠️  ORCID conflict (one pick is wrong) ({len(conflict)}): {', '.join(conflict) or '—'}")
    print(f"  low S2 confidence ({len(low)}): {', '.join(low) or '—'}")
    print(f"  no S2 profile ({len(missing_s2)}): {', '.join(missing_s2) or '—'}")
    print(f"  no OpenAlex profile ({len(missing_oa)}): {', '.join(missing_oa) or '—'}")
    print("\nVerify seeds.json, then: uv run build_graph.py")


if __name__ == "__main__":
    main()
