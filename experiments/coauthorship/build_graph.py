# /// script
# requires-python = ">=3.10"
# dependencies = ["httpx", "networkx", "numpy", "graspologic", "scikit-learn", "setuptools<81"]
# ///
"""Build the co-authorship graph JSON for /coauthorship/ from a hand-verified Semantic Scholar seed.

Identity is pinned by `seeds.json` (see build_seeds.py) — no fuzzy search, nothing guessed at build
time. For each seeded person we union their pinned S2 profile(s), fetch their papers, and build a
name-keyed co-authorship graph. For every pair of listed people we trace shortest paths (up to K_MAX
hops) and tag each node/edge with the hop at which it first appears (`minhop`) so the page can reveal
connecting people progressively. Detect communities (degree-corrected spectral + GMM), pre-compute a layout,
and write a static JSON the page loads directly (no live API calls at view time).

    cd experiments/coauthorship && uv run build_seeds.py   # once, then hand-verify seeds.json
    cd experiments/coauthorship && uv run build_graph.py   # rebuild the graph from verified seeds

Output: ../../assets/data/coauthorship.json. Raw S2 paper responses cached under raw/ (offline re-runs).
"""
import html
import json
import math
import os
import re

from overrides import apply_overrides, load_overrides
import time
import unicodedata
from collections import Counter, defaultdict
from itertools import islice
from pathlib import Path

import httpx
import networkx as nx
import numpy as np
from graspologic.embed import LaplacianSpectralEmbed
from sklearn.mixture import GaussianMixture

BASE = "https://api.semanticscholar.org/graph/v1"
OA_BASE = "https://api.openalex.org"
MAILTO = "alexloftus2004@gmail.com"          # OpenAlex polite pool
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
SEEDS = HERE / "seeds.json"
OVERRIDES = HERE / "overrides.json"   # post-build human/crowd corrections (see overrides.py)
RAW = HERE / "raw"
RAW.mkdir(exist_ok=True)
OUT = REPO / "assets" / "data" / "coauthorship.json"
# drop <slug>.jpg|png|webp here (slug = lowercase name, spaces -> hyphens) to fill a node's avatar
PHOTO_DIR = REPO / "assets" / "images" / "coauthors"

# connectors with these (common) names are dropped to avoid name-collision false bridges
COMMON_STOP = {
    "xin chen", "y liu", "jamie simon", "wei wang", "jun zhang", "wei li", "yang liu",
    "li wang", "jing wang", "yan zhang", "lei zhang", "jie chen", "wei zhang", "xin li",
}
# big papers (>MAX_AUTHORS) only contribute edges *between list members*, not edges through
# outside intermediaries. Under fractional weighting this is purely a TOPOLOGY guard, not a
# weight guard: hop reveal / path finding are unweighted BFS, so without it a single consortium
# paper (the corpus holds several at 100-509 authors) would mint hundreds of one-off connector
# nodes that read as legitimate 2-hop bridges. List<->list pairs on big papers still receive
# their full 1/n weight contribution either way.
MAX_AUTHORS = 25
# hop reveal: longest list<->list path (in edges) we trace, and how many shortest paths per pair
K_MAX = 4
PATHS_PER_PAIR = 4

# --- hand-verified audit corrections (independently cross-checked vs OpenAlex/ORCID/DBLP/arXiv) ---
# non-anchor fragments that carry a different name string than their twin, so name-keying alone can't
# fold them (anchors fold their own fragments via seeds.json `merge_ids`/`oa_merge_ids`).
NAME_ALIAS = {"c priebe": "carey e priebe"}    # "C. Priebe" -> "Carey E. Priebe"
# connectors whose single profile conflates two different real people (verified) — dropped as junk.
COLLISION_STOP = {
    "richard guo",   # JHU-stats R. Guo + Nomic-AI R. Guo merged -> false ronak mehta<->brandon duderstadt bridge
    "j smith",       # security J. Smith + ML J. Smith merged -> fabricated fred heiding<->stella biderman path
}
# anchor pairs with no real co-authorship (verified: no shared paper on OpenAlex/arXiv).
# merullo<->brinkmann comes only from a 49-author 2020 proceedings volume, not a joint paper.
EDGE_DROP = {tuple(sorted(("jack merullo", "jannik brinkmann")))}


def clean_display(s: str) -> str:
    """Normalize "Last, First" -> "First Last" so the two spellings key to one node."""
    if "," in s:
        parts = [p.strip() for p in s.split(",", 1)]
        if len(parts) == 2 and parts[1]:
            s = f"{parts[1]} {parts[0]}"
    return s


def clean_title(t: str) -> str:
    """Strip the HTML tags/entities OpenAlex (and occasionally S2) embed in titles — <sup>/<i>/<em>
    for formulae (e.g. "<sup>1</sup>H-MRS", "Fe<sub>2</sub>O<sub>3</sub>") and entities like &amp;.
    Left raw they show as literal markup in the popup AND split dedup (the "sup" tokens dodge the
    same-paper match), so the same work indexed by both sources can appear twice."""
    return html.unescape(re.sub(r"<[^>]+>", "", t or "")).strip()


def norm(s: str) -> str:
    s = clean_display(s)
    # Unicode hyphens/dashes -> ascii "-" FIRST: otherwise encode("ascii","ignore") deletes them with
    # no space ("Learning–based" -> "learningbased") while a plain "-" becomes a space, so the same
    # paper titled with an en-dash vs a hyphen normalizes differently and dodges the dedup.
    s = re.sub(r"[‐-―−]", "-", s)   # ‐ ‒ – — ― and minus sign -> "-"
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower().replace("-", " ")
    return " ".join(re.sub(r"[^\w\s]", " ", s).split())


def _main_title(title: str) -> str:
    """Normalized title before the first colon, but only if it's specific (>=3 tokens)."""
    head = norm(title.split(":", 1)[0])
    return head if len(head.split()) >= 3 else ""


def same_paper(t1: str, t2: str) -> bool:
    """Two records are the *same* paper if one title's tokens contain the other's (e.g. a
    "...Open-Weight..." or "Linearly ..." retitle) or they share a specific pre-colon main title
    (preprint/published versions with different subtitles). DOI/exact-title keying already merged the
    easy cases upstream; this catches the variant titles it can't. Applied per edge, so the
    false-merge blast radius is one pair's joint output."""
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
    """One representative title per distinct paper, collapsing version/variant duplicates."""
    clusters: list[str] = []                     # one representative title per distinct paper
    for t in titles:
        if not any(same_paper(t, rep) for rep in clusters):
            clusters.append(t)
    return clusters


def distinct_papers(titles: list[str]) -> int:
    """Count distinct papers in a list of titles, collapsing version/variant duplicates."""
    return len(distinct_title_list(titles))


def fractional_weight(records: list[tuple[str, int]]) -> float:
    """Fractional co-authorship weight: each distinct paper contributes 1/n_authors to its pairs
    (suggested by Stella Biderman; Newman 2001 uses 1/(n-1)), so a duo paper binds far tighter
    than a 30-author one. `records` = (title, n_authors); version variants collapse first
    (same_paper) and each cluster contributes 1/min(n) — the smallest author list among its
    versions, since an over-merged variant shouldn't dilute a genuinely small-team paper."""
    clusters: list[list[tuple[str, int]]] = []
    for t, n in records:
        for c in clusters:
            if same_paper(t, c[0][0]):
                c.append((t, n))
                break
        else:
            clusters.append([(t, n)])
    return sum(1.0 / max(1, min(n for _, n in c)) for c in clusters)


HEADERS = {"User-Agent": "coauthorship-build (alexloftus2004@gmail.com)"}
if os.environ.get("S2_API_KEY"):
    HEADERS["x-api-key"] = os.environ["S2_API_KEY"]
client = httpx.Client(timeout=60, headers=HEADERS)


def cached_get(client: httpx.Client, base: str, path: str, cache_key: str, **params) -> dict:
    """GET with disk cache + backoff on 429/network hiccup. Shared by the S2 and OpenAlex clients."""
    cache = RAW / f"{cache_key}.json"
    if cache.exists():
        return json.loads(cache.read_text())
    for attempt in range(8):
        try:
            r = client.get(f"{base}/{path}", params=params)
        except httpx.HTTPError:                          # transient network hiccup: back off, retry
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
    raise RuntimeError(f"giving up on {base}/{path} after repeated errors")


def get(path: str, cache_key: str, **params) -> dict:
    return cached_get(client, BASE, path, cache_key, **params)


def fetch_works(author_id: str) -> list[dict]:
    """All of one S2 author's papers, normalized to {id, doi, title, publication_year, citations,
    authorships}. (cache key bumped to _v2 when citationCount was added to the request.)"""
    out, offset = [], 0
    while True:
        data = get(f"author/{author_id}/papers", f"coauth_papers_v2_{author_id}_{offset}",
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


# ---- OpenAlex side (cross-reference / coverage union) -----------------------
oa_client = httpx.Client(timeout=60, headers={"User-Agent": f"coauthorship-build ({MAILTO})"})


def oa_bare(oa_url) -> str:
    return (oa_url or "").rstrip("/").split("/")[-1]


def oa_get(path: str, cache_key: str, **params) -> dict:
    params.setdefault("mailto", MAILTO)
    return cached_get(oa_client, OA_BASE, path, cache_key, **params)


def fetch_works_openalex(author_id: str) -> list[dict]:
    """All of one OpenAlex author's works, normalized to the same shape as fetch_works (S2)."""
    out, cursor, page = [], "*", 0
    while cursor:
        data = oa_get("works", f"oa_works_v2_{author_id}_{page}", filter=f"author.id:{author_id}",
                      per_page=200, cursor=cursor,
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


# ---- 1. load the hand-verified seed -----------------------------------------
# Each person is pinned to an S2 profile (always trusted) and, where it corroborates, an OpenAlex
# profile. The OpenAlex side is trusted for the paper union ONLY when crosscheck.agree (shared ORCID
# or shared top-cited paper) or the human explicitly pinned it (oa_verified) — this is the safety
# gate that keeps an over-merged OpenAlex homonym (e.g. a same-named geographer) out of the graph.
assert SEEDS.exists(), f"{SEEDS} missing — run `uv run build_seeds.py` first, then verify it"
seeds = json.loads(SEEDS.read_text())
seed_by_norm = {norm(e["name"]): e for e in seeds}   # canonical name -> seed entry (exclude_titles, …)
person_ids: dict[str, list[str]] = {}      # canonical name -> [pinned S2 author ids]
person_oa_ids: dict[str, list[str]] = {}   # canonical name -> [trusted OpenAlex author ids] (may be [])
unresolved: list[str] = []
oa_distrusted: list[str] = []              # has an OpenAlex pick the gate rejected (S2-only coverage)
# membership contract: every roster member needs a seeds entry (merge_affiliations writes
# stubs for self-joins; a hand-added core name needs a hand entry or a build_seeds run)
import registry as _registry
_roster_missing = ({norm(n) for n in _registry.load_roster()}
                   - {norm(e["name"]) for e in seeds})
assert not _roster_missing, f"roster members without a seeds entry: {sorted(_roster_missing)}"

for e in seeds:
    name = norm(e["name"])
    ids = [str(i) for i in ([e["s2_id"]] + e.get("merge_ids", [])) if i]
    person_ids[name] = ids
    if not ids:
        unresolved.append(e["name"])
    cc = e.get("crosscheck") or {}
    # OpenAlex is trusted when ORCID/top-paper agree or hand-pinned (oa_verified) — UNLESS oa_reject
    # is set, the override for a profile whose OpenAlex side over-merges a homonym (e.g. jeremy howard,
    # where OpenAlex injects a 1983 cheetah-biology paper and tourism articles into a 142-paper blob).
    trust_oa = bool(e.get("oa_verified") or cc.get("agree")) and not e.get("oa_reject")
    oa_ids = [oa_bare(i) for i in ([e.get("oa_id")] + e.get("oa_merge_ids", [])) if i] if trust_oa else []
    person_oa_ids[name] = oa_ids
    if e.get("oa_id") and not trust_oa:
        oa_distrusted.append(e["name"])
n_unverified = sum(1 for e in seeds if not e.get("verified"))
if n_unverified:
    print(f"⚠️  {n_unverified}/{len(seeds)} seeds not marked verified — building anyway "
          f"(set \"verified\": true in seeds.json once checked)")
if oa_distrusted:
    print(f"ℹ️  OpenAlex coverage withheld for {len(oa_distrusted)} (crosscheck disagrees / unpinned): "
          f"{', '.join(oa_distrusted)}\n")

resolved = [n for n in person_ids if person_ids[n]]

# ---- 2. fetch works from BOTH sources, collect aliases + a co-author index --
# Each work is tagged with its source ("s2"/"openalex") so the dedup below can record provenance.
print("Fetching works (Semantic Scholar + OpenAlex)...")
person_works: dict[str, list[dict]] = {}
alias_to_canon: dict[str, str] = {}        # normalized display name -> list-member canonical key
for q in resolved:
    works = []
    for aid in person_ids[q]:
        works += [dict(w, source="s2") for w in fetch_works(aid)]
    for aid in person_oa_ids.get(q, []):
        works += [dict(w, source="openalex") for w in fetch_works_openalex(aid)]
    # hand-curated exclude_titles: drop off-topic papers a same-named stranger contributed to an
    # otherwise-correct profile (S2/OpenAlex author disambiguation merges homonyms). Substring match
    # on the normalized title; applied before the union so strays affect neither edges nor the popup.
    excl = [norm(s) for s in (seed_by_norm.get(q, {}).get("exclude_titles") or []) if s]
    if excl:
        before = len(works)
        works = [w for w in works if not (w.get("title") and any(x in norm(w["title"]) for x in excl))]
        if before != len(works):
            print(f"    ↳ excluded {before - len(works)} stray paper(s) from {q}")
    person_works[q] = works
    alias_to_canon[q] = q
    s2set, oaset = set(person_ids[q]), set(person_oa_ids.get(q, []))
    for w in works:
        idset = s2set if w["source"] == "s2" else oaset
        for a in w["authorships"]:
            if str(a["author"].get("id")) in idset:     # an alias of q themselves -> map to q
                alias_to_canon[norm(a["author"]["display_name"])] = q
    n_s2 = len({w["id"] for w in works if w["source"] == "s2"})
    n_oa = len({w["id"] for w in works if w["source"] == "openalex"})
    print(f"  {q:22s} s2={n_s2:4} oa={n_oa:4}")
    time.sleep(0.02)


def canon(nm: str) -> str:
    """Map a normalized display name to its list-member canonical key (or itself)."""
    nm = NAME_ALIAS.get(nm, nm)                  # fold verified non-anchor fragments first
    return alias_to_canon.get(nm, nm)


def node_key(authorship: dict) -> str:
    return canon(norm(authorship["author"]["display_name"]))


# ---- 3. dedup papers across sources, build weighted co-authorship graph ------
# Dedup key = DOI, else normalized title, else source-local id. The SAME paper seen in both indices
# collapses to one record whose `sources` = {s2, openalex}; author lists are unioned by name (and we
# keep each author's S2 + OpenAlex id). Edges/nodes inherit the union of their papers' sources.
print("\nBuilding graph...")
list_nodes = set(resolved)


def paper_key(w: dict) -> str:
    if w.get("doi"):
        return "doi:" + w["doi"]
    if w.get("title"):
        return "t:" + norm(w["title"])
    return f'{w["source"]}:{w["id"]}'


papers: dict[str, dict] = {}               # key -> {"authors": {normname: {...}}, "sources": set}
for q in resolved:
    for w in person_works[q]:
        rec = papers.setdefault(paper_key(w), {"authors": {}, "sources": set(), "title": w.get("title", "")})
        rec["sources"].add(w["source"])
        if not rec["title"] and w.get("title"):
            rec["title"] = w["title"]
        for a in w["authorships"]:
            nm = norm(a["author"]["display_name"])
            if not nm:
                continue
            cur = rec["authors"].setdefault(nm, {"display_name": a["author"]["display_name"],
                                                 "s2_id": None, "oa_id": None})
            aid = a["author"].get("id")
            if aid and w["source"] == "s2" and not cur["s2_id"]:
                cur["s2_id"] = str(aid)
            if aid and w["source"] == "openalex" and not cur["oa_id"]:
                cur["oa_id"] = str(aid)

# per edge, (title, n_authors) of every shared record; weight = FRACTIONAL count after
# version-variant dedup: each distinct paper adds 1/n_authors (see fractional_weight)
edge_recs: defaultdict[tuple, list] = defaultdict(list)
edge_sources: defaultdict[tuple, set] = defaultdict(set)
node_sources: defaultdict[str, set] = defaultdict(set)
display_name: defaultdict[str, Counter] = defaultdict(Counter)
s2_id: defaultdict[str, Counter] = defaultdict(Counter)
oa_id: defaultdict[str, Counter] = defaultdict(Counter)


def add_pairs(keys, sources, title, n_auth):
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a, b = sorted((keys[i], keys[j]))
            edge_recs[(a, b)].append((title, n_auth))
            edge_sources[(a, b)] |= sources


for rec in papers.values():
    auths = list(rec["authors"].values())
    srcs = rec["sources"]
    keys = []
    for a in auths:
        k = canon(norm(a["display_name"]))
        keys.append(k)
        display_name[k][clean_display(a["display_name"])] += 1
        if a["s2_id"]:
            s2_id[k][a["s2_id"]] += 1
        if a["oa_id"]:
            oa_id[k][a["oa_id"]] += 1
        node_sources[k] |= srcs
    uniq = sorted(set(keys))
    list_in = [k for k in uniq if k in list_nodes]
    if len(auths) <= MAX_AUTHORS:
        add_pairs(uniq, srcs, rec["title"], len(auths))   # all co-author edges
    else:
        add_pairs(list_in, srcs, rec["title"], len(auths))  # big paper: only list<->list ties

# weight = fractional count over distinct papers (version/variant duplicates collapse first —
# preprint/published, "...Open-Weight..." retitles — then each paper adds 1/n_authors).
edge_w = {pair: round(fractional_weight(recs), 4) for pair, recs in edge_recs.items()}
for pair in EDGE_DROP:                         # verified non-co-authorships (proceedings-volume artifacts)
    edge_w.pop(pair, None)
    edge_sources.pop(pair, None)

G_full = nx.Graph()
for (u, v), w in edge_w.items():
    G_full.add_edge(u, v, weight=w)
for q in resolved:  # ensure isolated list members still exist as nodes
    G_full.add_node(q)

# drop junk intermediaries (common-name collisions, single-token names) before path search
junk = {n for n in G_full.nodes() if n not in list_nodes
        and (len(n.split()) < 2 or n in COMMON_STOP or n in COLLISION_STOP)}
G_full.remove_nodes_from(junk)

# ---- 4. hop reveal: shortest paths between listed people --------------------
# Each node/edge gets `minhop` = the length of the shortest list<->list path that first
# brings it in. The UI reveals everything with minhop <= the slider value:
#   hop 1 = direct co-authorships among listed people; hop k adds the outside people on
#   length-k paths between two listed people.
node_minhop = {n: (0 if n in list_nodes else math.inf) for n in G_full.nodes()}
edge_minhop = {}
listl = sorted(list_nodes)
pair_count = 0
for i, a in enumerate(listl):
    lengths = nx.single_source_shortest_path_length(G_full, a, cutoff=K_MAX)
    for b in listl[i + 1:]:
        L = lengths.get(b)
        if not L or L > K_MAX:
            continue
        pair_count += 1
        for path in islice(nx.all_shortest_paths(G_full, a, b), PATHS_PER_PAIR):
            for n in path:
                if n not in list_nodes and L < node_minhop[n]:
                    node_minhop[n] = L
            for u, v in zip(path, path[1:]):
                e = tuple(sorted((u, v)))
                if L < edge_minhop.get(e, math.inf):
                    edge_minhop[e] = L

kept = {n for n, h in node_minhop.items() if h < math.inf} | list_nodes
G = G_full.subgraph(kept).copy()
G = nx.Graph((u, v, d) for u, v, d in G.edges(data=True) if tuple(sorted((u, v))) in edge_minhop)
for n in kept:
    G.add_node(n)
print(f"  full: {G_full.number_of_nodes()} nodes (junk dropped: {len(junk)})")
print(f"  reveal graph: {G.number_of_nodes()} nodes / {G.number_of_edges()} edges"
      f" from {pair_count} connected list-pairs (<= {K_MAX} hops)")
hop_hist = Counter(min(h, K_MAX) for n, h in node_minhop.items() if n in kept)
print(f"  nodes by reveal hop: {dict(sorted(hop_hist.items()))}")

# ---- 5. communities: degree-corrected spectral embedding + GMM (Priebe / NeuroData) ----
# Regularized Laplacian spectral embedding (R-DAD) of the weighted reveal graph, row-normalized onto
# the unit sphere, then a Gaussian mixture with k = #named groups. The Laplacian (degree-normalized)
# embedding is what keeps EleutherAI and the Bau lab apart: on this graph they collaborate enough
# that plain adjacency (ASE) folds them into one blob, while LSE + row-norm recovers the three groups
# robustly across embedding dims 4-8. Runs on the largest connected component; isolated / fallback-
# routed list members fall outside it and stay uncolored (-1). GMM cluster ids are remapped to a
# fixed order keyed by *anchor person* so colours are stable across rebuilds.
COMMUNITY_LABELS = {            # anchor person (S2 spelling) -> (fixed id == colour, legend label)
    "stella biderman": (0, "EleutherAI"),
    "david bau": (1, "David Bau"),
    "j vogelstein": (2, "Joshua Vogelstein"),
}
EMBED_DIM = 7                   # Laplacian embedding dimension (pinned; was 6 — adding rohit
                                # gandikota's papers folded EleutherAI+Bau at dims 4-6; 7-8 separate
                                # all three anchors with sane clusters, checked 2026-06-12)
GMM_SEED = 7                    # GMM init seed (was 42 — adding giordano rogers tipped seed 42 into a
                                # degenerate optimum that merged Bau+Vogelstein at dim 7; seed 7 re-finds
                                # the established editorial clustering, sizes {El 24, Bau 56, Vog 29} ==
                                # the live structure, checked 2026-06-24)
# editorial overrides: force a person into the community of another anchor. The spectral+GMM model
# pulls some extended-interp people into EleutherAI on the strength of a couple of co-authorships;
# these reassign them by hand (value = anchor whose community to join). Applied after GMM labelling.
COMMUNITY_FORCE = {
    "kola ayonrinde": "david bau",   # extended-interp, not EleutherAI core (Alex's call)
    "jesse hoogland": "david bau",   # developmental interp / own thing; group with the interp cluster
}

node_comm = {n: -1 for n in G.nodes()}
cid, sizes, dim = 0, [], 0
if G.number_of_edges():
    lcc = sorted(max(nx.connected_components(G), key=len))
    A = nx.to_numpy_array(G.subgraph(lcc), nodelist=lcc, weight="weight")
    dim = min(EMBED_DIM, len(lcc) - 1)
    emb = LaplacianSpectralEmbed(n_components=dim, algorithm="full", form="R-DAD").fit_transform(A)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True).clip(1e-9)        # row-norm onto unit sphere
    k = len(COMMUNITY_LABELS)
    labels = GaussianMixture(k, covariance_type="full", random_state=GMM_SEED).fit_predict(emb)
    raw_of = dict(zip(lcc, (int(x) for x in labels)))
    remap = {raw_of[a]: i for a, (i, _) in COMMUNITY_LABELS.items() if a in raw_of}
    assert len(remap) == k, f"anchors did not separate into {k} clusters (got {remap}); model assumption broke"
    node_comm.update({n: remap[c] for n, c in raw_of.items()})
    cid = k
    for person, anchor in COMMUNITY_FORCE.items():     # editorial reassignments (see COMMUNITY_FORCE)
        if person in node_comm and node_comm.get(anchor, -1) >= 0:
            node_comm[person] = node_comm[anchor]
    sizes = [sum(v == i for v in node_comm.values()) for i in range(cid)]
print(f"  communities (LSE+GMM, dim={dim}): {cid} groups, sizes {sizes}")

# ---- 6. pre-computed layout (whole reveal graph; positions stay fixed) ------
pos = nx.spring_layout(G, weight="weight", seed=42, k=1.3, iterations=250)

# ---- 6b. bridge crawl: route isolated list members into the network ---------
# The base graph only covers the listed people's *own* papers, so anyone whose co-authors never
# appear on someone else's paper lands in their own component (e.g. Roy Rinberg). To find a real
# co-authorship route we crawl outward from those islands: fetch the islands' intermediate authors'
# OTHER papers (which reveal their wider collaborations) until each isolated person reaches the main
# component, or we exhaust a fetch budget. Only the resulting path nodes/edges are emitted; the rest
# of the expansion is discarded and the main reveal graph G is never touched.
BRIDGE_FETCH_BUDGET = int(os.environ.get("BRIDGE_BUDGET", "800"))   # max intermediate-author fetches
g_nodes = set(G.nodes())                                            # frozen: never relabel these
connected_list = {n for n in list_nodes if G.degree(n) > 0}
isolated_list = [n for n in sorted(list_nodes) if G.degree(n) == 0]


def add_paper_edges(graph, authorships):
    """Add a fetched paper's co-author edges to `graph`; learn labels/ids for NEW authors only."""
    keys = []
    for a in authorships:
        k = node_key(a)
        keys.append(k)
        if k not in g_nodes:                       # don't perturb the committed graph's labels
            display_name[k][clean_display(a["author"]["display_name"])] += 1
            aid = a["author"].get("id")
            if aid:
                s2_id[k][str(aid)] += 1
    uniq = sorted(set(keys))
    pairs = uniq if len(authorships) <= MAX_AUTHORS else [k for k in uniq if k in list_nodes]
    w = 1.0 / max(1, len(authorships))         # same fractional rule as the committed graph
    for i in range(len(pairs)):
        for j in range(i + 1, len(pairs)):
            a, b = pairs[i], pairs[j]
            if a == b or tuple(sorted((a, b))) in EDGE_DROP:   # verified non-collaboration: never re-add
                continue
            if graph.has_edge(a, b):
                graph[a][b]["weight"] += w
            else:
                graph.add_edge(a, b, weight=w)


def pruned(graph):
    # bridge-finding is stricter than the base junk filter: also drop initials-form names
    # ("j smith", "b schneier") — name-keying collapses every such profile into one false hub.
    junky = {n for n in graph if n not in list_nodes
             and (len(n.split()) < 2 or n in COMMON_STOP or n in COLLISION_STOP or len(n.split()[0]) == 1)}
    H = graph.copy()
    H.remove_nodes_from(junky)
    return H


def author_id_of(name):
    return s2_id[name].most_common(1)[0][0] if name in s2_id else None


GX = G_full.copy()                                   # working graph we grow for path-finding only
fetched = {x for q in resolved for x in person_ids[q]}
H = pruned(GX)
main = max(nx.connected_components(H), key=len) if H else set()
pending = [u for u in isolated_list if u in H and u not in main]
fetch_count, hop = 0, 0
while pending and fetch_count < BRIDGE_FETCH_BUDGET:
    hop += 1
    frontier = sorted({a for u in pending for a in nx.node_connected_component(H, u)
                       if a not in list_nodes and author_id_of(a) and author_id_of(a) not in fetched})
    if not frontier:
        break
    for name in frontier:
        if fetch_count >= BRIDGE_FETCH_BUDGET:
            break
        aid = author_id_of(name)
        if aid in fetched:
            continue
        fetched.add(aid)
        fetch_count += 1
        try:                                  # best-effort: a rate-limited intermediary just doesn't route
            works = fetch_works(aid)
        except RuntimeError:
            continue
        for w in works:
            add_paper_edges(GX, w["authorships"])
    H = pruned(GX)
    main = max(nx.connected_components(H), key=len)
    pending = [u for u in isolated_list if u in H and u not in main]
    print(f"  bridge hop {hop}: {fetch_count} authors fetched, "
          f"{len(isolated_list) - len(pending)}/{len(isolated_list)} routed, GX={H.number_of_nodes()} nodes")

# shortest path from each (now-reachable) isolated member to the nearest connected listed author
fallback_paths = {}                                  # isolated id -> [path node ids]
for u in isolated_list:
    if u not in H:
        continue
    lengths = nx.single_source_shortest_path_length(H, u)
    cand = [(L, -G.degree(t), t) for t, L in lengths.items()
            if t in connected_list and t != u]       # nearest connected author, prefer high-degree
    if cand:
        fallback_paths[u] = nx.shortest_path(H, u, min(cand)[2])
print(f"  fallback paths: {len(fallback_paths)}/{len(isolated_list)} isolated people routed to an author "
      f"({fetch_count} bridge fetches)")

# connector seeds: interpolate each path-only node along the line from its isolated person to the
# author it reaches. The committed graph's positions stay exactly as computed above; the page's
# live force layout re-packs everything anyway, so these are only starting points.
path_only, seen_po = [], set()
for p in fallback_paths.values():
    (x0, y0), (x1, y1) = pos[p[0]], pos[p[-1]]
    for i, n in enumerate(p):
        if n in G or n in seen_po:
            continue
        seen_po.add(n)
        path_only.append(n)
        f = i / (len(p) - 1)
        pos[n] = (x0 * (1 - f) + x1 * f, y0 * (1 - f) + y1 * f)

# ---- 7. assemble nodes/links ------------------------------------------------
# Force a node's display name when every spelling S2/OpenAlex carries is an abbreviation. Antonio
# Mari's lone in-network paper (the SAE/diffusion co-event with Wendler) lists him as "A. Mari", so
# the most-common-spelling rule would label him "A. Mari"; pin his full name instead.
LABEL_OVERRIDE = {"antonio mari": "Antonio Mari"}


def nice_label(key):
    if key in LABEL_OVERRIDE:
        return LABEL_OVERRIDE[key]
    return display_name[key].most_common(1)[0][0] if display_name[key] else key.title()

def initials(label):
    parts = [p for p in re.split(r"\s+", label) if p and p[0].isalpha()]
    if not parts:
        return label[:2].upper()
    return (parts[0][0] + (parts[-1][0] if len(parts) > 1 else "")).upper()

def profile_url(key):
    if key in list_nodes and person_ids.get(key):
        return f"https://www.semanticscholar.org/author/{person_ids[key][0]}"
    if s2_id[key]:
        return f"https://www.semanticscholar.org/author/{s2_id[key].most_common(1)[0][0]}"
    return None

def photo_url(key):
    slug = key.replace(" ", "-")
    for ext in ("jpg", "jpeg", "png", "webp"):
        if (PHOTO_DIR / f"{slug}.{ext}").exists():
            return f"/assets/images/coauthors/{slug}.{ext}"
    return None

def oa_profile_url(key):
    return f"https://openalex.org/{oa_id[key].most_common(1)[0][0]}" if oa_id[key] else None

DEFAULT_SRC = {"s2"}     # defensive fallback if a node/edge somehow recorded no source

def src_tag(sources: set) -> str:
    """{s2, openalex} -> 'both'; single source -> 's2' / 'oa'. Drives the page's provenance UI."""
    if {"s2", "openalex"} <= sources:
        return "both"
    return "oa" if sources == {"openalex"} else "s2"

def papers_for(key):
    """A list member's own papers for the hover popup, most-cited first. Collapses duplicates with the
    SAME fuzzy match used for edge weights (`same_paper`) — not just exact DOI/title — so version
    variants (arXiv vs published, retitles) and the same paper indexed by BOTH S2 and OpenAlex (which
    often disagree on DOI) fold into one row. Each cluster keeps its best-cited representative title;
    citations are the max across the merged records (S2 and OpenAlex count differently)."""
    clusters = []                                # [{title, year, cites}] — one per distinct paper
    for w in person_works.get(key, []):
        title = w.get("title")
        if not title:
            continue
        cites, year = w.get("citations") or 0, w.get("publication_year")
        hit = next((c for c in clusters if same_paper(title, c["title"])), None)
        if hit is None:
            clusters.append({"title": title, "year": year, "cites": cites})
        else:
            if cites > hit["cites"]:             # keep the higher-cited record's title as the rep
                hit["title"] = title
            hit["cites"] = max(hit["cites"], cites)
            if hit["year"] is None:
                hit["year"] = year
    clusters.sort(key=lambda c: (-c["cites"], c["year"] is None, -(c["year"] or 0), c["title"]))
    return clusters

# Papers each node shares with at least one OTHER node shown here. Robust to profile-merge
# inflation: a list member whose cluster accidentally merged same-named strangers still only
# counts papers co-authored with someone actually in this network.
shared_titles: defaultdict[str, list] = defaultdict(list)
for rec in papers.values():
    on = {canon(norm(a["display_name"])) for a in rec["authors"].values()} & kept
    if len(on) >= 2:
        for k in on:
            shared_titles[k].append(rec["title"])
shared_papers = {k: distinct_papers(titles) for k, titles in shared_titles.items()}  # version-deduped

nodes = []
for n in G.nodes():
    x, y = pos[n]
    nodes.append({
        "id": n,
        "label": nice_label(n),
        "initials": initials(nice_label(n)),
        "is_list": n in list_nodes,
        "minhop": int(min(node_minhop[n], K_MAX)),
        "shared_papers": shared_papers.get(n, 0),
        "degree": G.degree(n),
        "community": node_comm[n],
        "x": round(float(x), 4),
        "y": round(float(y), 4),
        "openalex": profile_url(n),   # kept key name for the JS; now points at Semantic Scholar
        "oa_url": oa_profile_url(n),  # real OpenAlex profile, when this node has one
        "sources": src_tag(node_sources.get(n, DEFAULT_SRC)),
        "photo": photo_url(n),
        "no_papers": False,
        "papers": papers_for(n) if n in list_nodes else [],
    })
nodes.sort(key=lambda d: (not d["is_list"], d["minhop"], -d["degree"]))

# path-only connector avatars (kept out of the slider: minhop > K_MAX, never hop-revealed)
for n in path_only:
    x, y = pos[n]
    nodes.append({
        "id": n,
        "label": nice_label(n),
        "initials": initials(nice_label(n)),
        "is_list": False,
        "minhop": K_MAX + 1,
        "shared_papers": 0,
        "degree": 1,                  # small connector avatar
        "community": -1,
        "x": round(float(x), 4),
        "y": round(float(y), 4),
        "openalex": profile_url(n),
        "oa_url": oa_profile_url(n),
        "sources": src_tag(node_sources.get(n, DEFAULT_SRC)),
        "photo": photo_url(n),
        "path_only": True,
        "no_papers": False,
        "papers": [],
    })

# listed people with no indexed papers at all (no S2 profile). They have no edges, so they ride
# the live layout like the isolated members; the page draws them as open circles (a distinct colour
# from has-papers-but-unconnected people) and they carry no paper popup.
no_paper_nodes = [norm(name) for name in unresolved]
for i, key in enumerate(no_paper_nodes):
    ang = 2 * math.pi * i / max(1, len(no_paper_nodes))
    x, y = 1.3 * math.cos(ang), 1.3 * math.sin(ang)
    label = nice_label(key)               # empty display_name -> key.title(); keeps all builders consistent
    nodes.append({
        "id": key,
        "label": label,
        "initials": initials(label),
        "is_list": True,
        "minhop": 0,                  # a listed person; never edge-revealed (degree 0)
        "shared_papers": 0,
        "degree": 0,
        "community": -1,
        "x": round(float(x), 4),
        "y": round(float(y), 4),
        "openalex": None,
        "oa_url": None,
        "sources": "s2",              # no papers in either index; tag kept for a uniform schema
        "photo": photo_url(key),
        "no_papers": True,
        "papers": [],
    })

def edge_papers(key):  # deduped, display-ready shared-paper titles for an edge (blanks dropped)
    return [t for t in distinct_title_list([t for t, _ in edge_recs.get(key, [])]) if t]

links = [{"source": u, "target": v, "weight": d.get("weight", 1),
          "n_papers": len(edge_papers(tuple(sorted((u, v))))),   # integer count for display
          "minhop": int(edge_minhop[tuple(sorted((u, v)))]),
          "sources": src_tag(edge_sources.get(tuple(sorted((u, v))), DEFAULT_SRC)),
          "papers": edge_papers(tuple(sorted((u, v))))}
         for u, v, d in G.edges(data=True)]

# fallback-path edges (one object per unique pair, shared across paths) + the routes themselves
pair_link = {}
for p in fallback_paths.values():
    for a, b in zip(p, p[1:]):
        key = tuple(sorted((a, b)))
        pair_link.setdefault(key, {"source": key[0], "target": key[1],
                                   "weight": round(GX[a][b].get("weight", 1), 4),
                                   "papers": edge_papers(key)})
path_links = list(pair_link.values())
paths = {u: {"path": p, "target": p[-1], "len": len(p) - 1} for u, p in fallback_paths.items()}

# legend label per community: the anchor whose fixed id is this community (present for ids 0..cid-1),
# else the highest-degree member as a fallback.
communities = []
for c in range(cid):
    members = [n for n in node_comm if node_comm[n] == c]
    forced = next((lbl for a, (i, lbl) in COMMUNITY_LABELS.items() if i == c), None)
    label = forced or (nice_label(max(members, key=G.degree)) if members else f"group {c}")
    communities.append({"id": c, "label": label})

# list members that never connect within K_MAX hops (shown only in the side list)
unconnected = sorted(nice_label(n) for n in list_nodes if G.degree(n) == 0)

# every listed person — including the no-paper people — must appear as a node now
all_listed = {norm(e["name"]) for e in seeds}

# ---- sanity asserts ---------------------------------------------------------
node_ids = {d["id"] for d in nodes}
assert G.number_of_nodes() > 0, "empty graph"
assert all(n in node_ids for n in list_nodes), "missing list member"
assert all(k in node_ids for k in all_listed), "a listed person is missing from the graph"
assert len([d for d in nodes if d["no_papers"]]) == len(unresolved), "no-paper node count mismatch"
assert all(d["x"] == d["x"] and d["y"] == d["y"] for d in nodes), "NaN positions"
assert all(l["minhop"] >= 1 for l in links), "edge without hop level"
assert 40 <= len(nodes) <= 600, f"node count out of expected range: {len(nodes)}"
assert all(n in node_ids for p in fallback_paths.values() for n in p), "path node missing from nodes"
assert all(p[0] in list_nodes and p[-1] in connected_list for p in fallback_paths.values()), "bad path endpoints"

out = {
    "nodes": nodes,
    "links": links,
    "path_links": path_links,
    "paths": paths,
    "communities": communities,
    "unresolved": [q.title() for q in unresolved],
    "unconnected": unconnected,
    "meta": {
        "source": "Semantic Scholar + OpenAlex",
        "k_max": K_MAX,
        "n_nodes": len(nodes),
        "n_links": len(links),
        "n_list": len(all_listed),
        "n_no_papers": len(unresolved),
        "n_communities": cid,
        "weighting": "fractional",   # each distinct paper adds 1/n_authors per pair (Newman-style)
        "links_by_source": dict(Counter(l["sources"] for l in links)),
        "n_oa_trusted": sum(1 for v in person_oa_ids.values() if v),
    },
}
# replay human/crowd corrections (overrides.json) onto the finished graph — the last build step,
# so they survive every nightly rebuild without perturbing the graph/hop/clustering algorithms above.
ov = load_overrides(OVERRIDES)
out = apply_overrides(out, ov)
n_ov = sum(len(ov.get(k, [])) for k in ("remove_nodes", "remove_papers", "add_papers", "remove_edges")) \
    + sum(len(ov.get(k, {})) for k in ("node_label", "node_community", "node_url", "node_photo", "paper_rename"))
if n_ov:
    print(f"applied {n_ov} override(s) from {OVERRIDES.name}")

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(out, indent=2))
print(f"\nwrote {OUT}  ({len(nodes)} nodes, {len(links)} links, {cid} communities, "
      f"{len(paths)} fallback paths / {len(path_only)} connectors)")
print(f"unresolved ({len(unresolved)}): {', '.join(unresolved)}")
