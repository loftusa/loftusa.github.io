"""The person registry: roster.json is the single source of truth for WHO is on the maps.

THE CONTRACT (binding on every current and future network over this node set):
a network's build derives its membership from this registry, not from its own data file.
A person in the roster with no data in some network renders as that network's honest
"empty" state (hollow node on the papers map, zero chapters on the careers map, a sparse
seat in the analyses) — never as an absence. That is what makes "add a person once,
they appear everywhere" true by construction, including for networks built later.

roster.json: {"core": [names Alex curates], "self_joined": [names owned by
merge_affiliations.py — added/removed as join events come and go]}. Names are source-style
keys ("leo mckee-reid"); join on other datasets via norm_person.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from affiliation_events import norm_person  # noqa: E402

ROSTER_PATH = Path(__file__).resolve().parent / "roster.json"


def load_roster(path: Path = ROSTER_PATH) -> list[str]:
    """All member names (core + self_joined), duplicate-checked."""
    r = json.loads(path.read_text())
    names = r["core"] + r["self_joined"]
    assert len({norm_person(n) for n in names}) == len(names), "duplicate person in roster"
    return names


def reconcile_membership(records: dict, path: Path = ROSTER_PATH,
                         empty_record=None) -> tuple[dict, list[str]]:
    """Make a network's person-records dict match the registry exactly.

    Roster people missing from `records` get `empty_record` (a fresh dict per person);
    records for people NOT in the roster are a hard error (the registry is authoritative —
    add the person to roster.json, don't smuggle them in through a data file).
    Returns (records, names_added). Mutates and returns `records` for convenience.
    """
    names = load_roster(path)
    norms = {norm_person(n): n for n in names}
    extra = sorted(set(map(norm_person, records)) - set(norms))
    assert not extra, f"people in network data but not in roster.json: {extra}"
    added = []
    have = set(map(norm_person, records))
    for n in names:
        if norm_person(n) not in have:
            records[n.lower()] = (empty_record() if callable(empty_record)
                                  else {"entries": [], "city": "", "identity_confident": True,
                                        "notes": "roster member — no recorded chapters yet",
                                        "reviewed": False})
            added.append(n)
    return records, added
