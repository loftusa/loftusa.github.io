"""Affiliation self-service events: fold the append-only log, apply the overlay at build time.

Sibling of overrides.py, same philosophy for the affiliation graph: people POST events from the
site (edit own row / join the map — see chat_api.py /affiliations/*), the log is folded (pure,
chronological last-write-wins, ts-aware set/remove undo) into a small overlay, and the overlay is
merged IN MEMORY over the hand-curated affiliations.json at build time (`apply_aff_overlay`).
The source file is never machine-written, so deleting a bad event from the log regenerates
everything clean on the next nightly (merge_affiliations.py).

Event types (payloads validated at POST time in chat_api.py):
  aff_entry_set    {person, org, type, role?, years?, current?, source?}   upsert own entry
  aff_entry_remove {person, org}                                           remove own entry
  aff_city         {person, city}                                          "" clears
  aff_join         {name, city?, entries?[<=10], scholar_url?, homepage?}  new person
  aff_confirm      {person}                                                "my row is correct"

Keys: person/name fold with `norm_person` (the graph-id normalization: punctuation -> space);
org strings fold with `org_key` (whitespace-collapse + casefold — parentheticals distinguish
real orgs, so no punctuation stripping). Org strings stay RAW: canonicalization (CANON) remains
a build-time concern; `apply_aff_overlay` only does spelling *adoption* to avoid slug collisions.
"""
from __future__ import annotations

import copy
import json
import re
from pathlib import Path

ENTRY_TYPES = {"lab", "program", "company", "university", "community"}

AFF_EMPTY: dict = {
    "version": 1,
    "entry_set": {},      # pid -> {org_key -> {org, type, role, years, current, source, ts}}
    "entry_remove": {},   # pid -> [org_key, ...]
    "city": {},           # pid -> str ("" clears)
    "join": {},           # pid -> {name, city, scholar_url, homepage, ts}
    "confirmed": {},      # pid -> ts of the latest "my row is correct"
}

MAX_ENTRIES_PER_PERSON = 12

# per-field length caps — single home; the API validates against these (422), the fold
# truncates as a backstop for events that predate a cap change
FIELD_CAPS = {"person": 80, "name": 80, "org": 200, "role": 200, "years": 40,
              "city": 80, "source": 300, "scholar_url": 300, "homepage": 300}


def norm_person(s: str) -> str:
    """Graph-id normalization (must match build_affiliations.norm): punctuation folds to space."""
    return re.sub(r"[^a-z0-9]+", " ", str(s).lower()).strip()


def org_key(s: str) -> str:
    """Fold key for org strings: whitespace-collapse + casefold, punctuation preserved."""
    return " ".join(str(s).split()).casefold()


def _clean_entry(spec: dict, ts: str) -> dict | None:
    """Validate + normalize one entry payload into the overlay entry shape. None = drop."""
    org = " ".join(str(spec.get("org", "")).split())
    typ = spec.get("type")
    if not org or len(org) > FIELD_CAPS["org"] or typ not in ENTRY_TYPES:
        return None
    return {
        "org": org,
        "type": typ,
        "role": str(spec.get("role", "") or "")[:FIELD_CAPS["role"]],
        "years": str(spec.get("years", "") or "")[:FIELD_CAPS["years"]],
        "current": bool(spec.get("current", False)),
        "source": str(spec.get("source", "") or "")[:FIELD_CAPS["source"]],
        "ts": ts,
    }


def fold_aff_events(events: list[dict]) -> dict:
    """Fold the event log into the overlay contract (pure; idempotent; same log -> same dict).

    Chronological processing gives natural LWW; per-(person, org_key) a later remove suppresses
    an earlier set and vice versa (the mirror of overrides.py's add/remove paper resolution).
    Invalid entry types and unknown event types are dropped (forward-compatible).
    """
    out = copy.deepcopy(AFF_EMPTY)
    removed: dict[str, dict[str, str]] = {}          # pid -> {org_key: ts of latest remove}
    for e in sorted(events, key=lambda e: e.get("ts", "")):
        p, ts = e.get("payload", {}), e.get("ts", "")
        kind = e.get("type")
        if kind == "aff_join":
            pid = norm_person(p.get("name", ""))
            if not pid:
                continue
            out["join"][pid] = {
                "name": " ".join(str(p.get("name", "")).split())[:80],
                "city": str(p.get("city", "") or "")[:80],
                "scholar_url": (str(p.get("scholar_url") or "")[:300] or None),
                "homepage": (str(p.get("homepage") or "")[:300] or None),
                "ts": ts,
            }
            if p.get("city"):
                out["city"][pid] = str(p["city"])[:80]
            for spec in (p.get("entries") or [])[:10]:
                entry = _clean_entry(spec, ts)
                if entry:
                    out["entry_set"].setdefault(pid, {})[org_key(entry["org"])] = entry
                    removed.get(pid, {}).pop(org_key(entry["org"]), None)
        elif kind == "aff_entry_set":
            pid = norm_person(p.get("person", ""))
            entry = _clean_entry(p, ts)
            if not pid or not entry:
                continue
            out["entry_set"].setdefault(pid, {})[org_key(entry["org"])] = entry
            removed.get(pid, {}).pop(org_key(entry["org"]), None)
        elif kind == "aff_entry_remove":
            pid, okey = norm_person(p.get("person", "")), org_key(p.get("org", ""))
            if not pid or not okey:
                continue
            out["entry_set"].get(pid, {}).pop(okey, None)
            removed.setdefault(pid, {})[okey] = ts
        elif kind == "aff_city":
            pid = norm_person(p.get("person", ""))
            if pid and "city" in p:
                out["city"][pid] = str(p["city"] or "")[:80]
        elif kind == "aff_confirm":
            pid = norm_person(p.get("person", ""))
            if pid:
                out["confirmed"][pid] = ts
        # unknown types: ignored
    out["entry_set"] = {pid: m for pid, m in out["entry_set"].items() if m}
    out["entry_remove"] = {pid: sorted(m) for pid, m in removed.items() if m}
    return out


def load_aff_overrides(path: Path) -> dict:
    """The committed overlay, or the empty contract when absent (mirror of load_overrides)."""
    return json.loads(path.read_text()) if path.exists() else copy.deepcopy(AFF_EMPTY)


def _org_slug(s: str) -> str:
    """The build's org node id — vendored from build_affiliations.slug (KEEP IN SYNC; importing
    would be circular). Adoption must dedupe in slug space because that's where the build's
    collision assert lives."""
    import unicodedata
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")


def apply_aff_overlay(src: dict, overlay: dict, canon: dict | None = None,
                      ) -> tuple[dict, list[str]]:
    """Merge the folded overlay over the hand-curated source records, in memory (pure).

    Returns (merged, warnings). The source dict is never mutated. All org matching happens in
    CANONICAL space: the UI shows canonical labels while the base file stores raw variants, so an
    incoming org_key must match an entry either by its raw spelling or by its CANON label. New
    org strings are ADOPTED onto an existing spelling when they collide in org_key OR slug space
    (the build asserts one label per slug), with the established type locked; genuinely new orgs
    pass through raw, surface as their own node, and emit a "needs a CANON line" warning — that
    warning stream is the human review queue.
    """
    canon = canon or {}
    merged = copy.deepcopy(src)
    warnings: list[str] = []
    pid_of = {norm_person(k): k for k in merged}

    def canon_label(org: str) -> str:
        return canon[org][0] if org in canon else org

    def entry_keys(e: dict) -> set[str]:
        return {org_key(e["org"]), org_key(canon_label(e["org"]))}

    for pid in sorted(overlay.get("join", {})):
        j = overlay["join"][pid]
        if pid in pid_of:
            warnings.append(f"join '{j['name']}' collides with existing person — skipped")
            continue
        key = " ".join(j["name"].split()).lower()
        extras = " · ".join(filter(None, [j.get("scholar_url"), j.get("homepage")]))
        merged[key] = {
            "entries": [],
            "city": j.get("city", ""),
            "identity_confident": True,
            "notes": f"self-joined via site ({j['ts'][:10]})" + (f" — {extras}" if extras else ""),
            "reviewed": False,
        }
        pid_of[pid] = key

    for pid in sorted(overlay.get("entry_remove", {})):
        key = pid_of.get(pid)
        if key is None:
            continue                                  # e.g. edits orphaned by a join revert
        gone = set(overlay["entry_remove"][pid])
        merged[key]["entries"] = [e for e in merged[key]["entries"]
                                  if not (entry_keys(e) & gone)]

    # adoption pool, kept LIVE as entries append so two new spellings of the same org converge
    # on the first one (deterministic: pids and org keys iterate sorted). Indexed by org_key AND
    # slug; values (label, locked type). setdefault order: canon labels (authoritative types),
    # canon raw keys (hand spellings, canon's type), then existing entries.
    pool_key: dict[str, tuple[str, str | None]] = {}
    pool_slug: dict[str, tuple[str, str | None]] = {}

    def pool_add(label: str, typ: str | None) -> None:
        pool_key.setdefault(org_key(label), (label, typ))
        pool_slug.setdefault(_org_slug(label), (label, typ))

    for label, typ in {lbl: t for (lbl, t) in canon.values()}.items():
        pool_add(label, typ)
    for raw, (_, typ) in canon.items():
        pool_add(raw, typ)
    for rec in merged.values():
        for e in rec["entries"]:
            pool_add(e["org"], canon[e["org"]][1] if e["org"] in canon else e["type"])

    for pid in sorted(overlay.get("entry_set", {})):
        key = pid_of.get(pid)
        if key is None:
            warnings.append(f"entry edits for unknown person '{pid}' — skipped")
            continue
        entries = merged[key]["entries"]
        for okey in sorted(overlay["entry_set"][pid]):
            spec = overlay["entry_set"][pid][okey]
            provenance = spec["source"] or f"self-reported via site ({spec['ts'][:10]})"
            existing = next((e for e in entries if okey in entry_keys(e)), None)
            if existing is not None:                  # keep the base org spelling AND its type
                existing.update(role=spec["role"], years=spec["years"],
                                current=spec["current"], source=provenance, verified=False)
                continue
            if len(entries) >= MAX_ENTRIES_PER_PERSON:
                warnings.append(f"{key}: entry cap reached — dropped '{spec['org']}'")
                continue
            hit = pool_key.get(okey) or pool_slug.get(_org_slug(spec["org"]))
            if hit:
                adopted, locked = hit
            else:
                adopted, locked = spec["org"], None
                warnings.append(f"new org '{spec['org']}' (from {key}) — needs a CANON line "
                                "if it's a variant of an existing org")
            typ = locked or spec["type"]
            entries.append({
                "org": adopted, "type": typ,
                "role": spec["role"], "years": spec["years"], "current": spec["current"],
                "source": provenance, "evidence": "", "verified": False,
            })
            pool_add(adopted, typ)

    for pid in sorted(overlay.get("city", {})):
        key = pid_of.get(pid)
        if key is not None:
            merged[key]["city"] = overlay["city"][pid]

    return merged, warnings


def make_seed_stub(name: str, homepage: str | None = None) -> dict:
    """A hand-checked-shape 'no profiles' seeds.json entry for a self-joined member (the ted-kyi
    pattern: null pins + verified flags keep the auto-pickers off it; `self_joined` marks it as
    machine-owned so merge_affiliations.py may add/remove it — it never touches untagged seeds)."""
    return {
        "name": name, "s2_id": None, "url": None, "affiliation": "", "homepage": homepage,
        "orcid": None, "n_papers": None, "h_index": None, "confidence": "none", "verified": True,
        "sample_titles": [], "merge_ids": [], "alternatives": [],
        "oa_id": None, "oa_url": None, "oa_affiliation": "", "oa_n_works": None, "oa_orcid": None,
        "oa_titles": [], "oa_merge_ids": [], "oa_alternatives": [],
        "crosscheck": {"orcid_s2": None, "orcid_oa": None, "orcid_match": False,
                       "orcid_conflict": False, "title_overlap_n": 0, "title_overlap_frac": 0.0,
                       "agree": False},
        "oa_verified": True, "self_joined": True,
    }
