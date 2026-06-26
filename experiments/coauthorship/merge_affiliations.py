# /// script
# requires-python = ">=3.10"
# dependencies = ["click", "httpx"]
# ///
"""Pull the affiliation self-service overlay into affiliation_overrides.json and sync the
self-joined seed stubs. Run nightly BEFORE build_graph.py (a join's stub must exist when the
graph builds) — see merge_corrections.py for the sister flow.

    cd experiments/coauthorship
    uv run merge_affiliations.py             # live overlay (open endpoint, no token)
    uv run merge_affiliations.py --dry-run   # print what would change, write nothing
    uv run merge_affiliations.py --from events.jsonl   # offline: fold a local event log

Fail-safe: any fetch failure warns and leaves every file untouched (the nightly proceeds).
Regenerative: affiliation_overrides.json is rewritten wholesale from the event log, and the
seeds/roster sync only ever adds or removes entries tagged `"self_joined": true` — deleting a
join event via the API's admin DELETE makes the next run remove that person everywhere.
"""
import json
import sys
from pathlib import Path

import click

sys.path.insert(0, str(Path(__file__).resolve().parent))
from affiliation_events import AFF_EMPTY, fold_aff_events, make_seed_stub, norm_person  # noqa: E402
from merge_corrections import fetch, parse_jsonl  # noqa: E402  (shared fail-safe fetch)

HERE = Path(__file__).resolve().parent
OVERRIDES = HERE / "affiliation_overrides.json"
SEEDS = HERE / "seeds.json"
ROSTER = HERE / "roster.json"
DEFAULT_API = "https://llm-resume-restless-thunder-9259.fly.dev"


def sync_stubs(overlay: dict, seeds: list[dict], roster: dict) -> tuple[list[str], list[str]]:
    """Mutate seeds + roster so the `self_joined`-tagged subset exactly mirrors the overlay's
    joins (untagged entries are never touched). Returns (added names, removed names)."""
    unmanaged = {norm_person(s["name"]) for s in seeds if not s.get("self_joined")}
    want = {pid: j for pid, j in overlay.get("join", {}).items() if pid not in unmanaged}

    removed = [s["name"] for s in seeds
               if s.get("self_joined") and norm_person(s["name"]) not in want]
    seeds[:] = [s for s in seeds if not (s.get("self_joined")
                                         and norm_person(s["name"]) not in want)]
    have = {norm_person(s["name"]) for s in seeds}
    added = []
    for pid in sorted(set(want) - have):
        j = want[pid]
        name = j["name"].strip().lower()
        seeds.append(make_seed_stub(name, homepage=j.get("homepage")))
        added.append(name)

    roster["self_joined"] = sorted({s["name"] for s in seeds if s.get("self_joined")})
    return added, removed


@click.command()
@click.option("--api", default=DEFAULT_API, show_default=True, envvar="COAUTHOR_API",
              help="API base URL.")
@click.option("--from", "from_file", type=click.Path(exists=True),
              help="Fold a local JSONL event log instead of fetching (offline/testing).")
@click.option("--dry-run", is_flag=True, help="Print the result; write nothing.")
def main(api: str, from_file: str | None, dry_run: bool) -> None:
    if from_file:
        events = parse_jsonl(Path(from_file).read_text())
        overlay, source = fold_aff_events(events), f"{len(events)} events ({from_file})"
    else:
        r = fetch(f"{api.rstrip('/')}/affiliations/overlay", {})
        if r is None:
            return                                   # fail-safe: warned, nothing touched
        overlay, source = {**AFF_EMPTY, **r.json()}, "live overlay"

    seeds = json.loads(SEEDS.read_text())
    roster = json.loads(ROSTER.read_text())
    added, removed = sync_stubs(overlay, seeds, roster)

    n_edits = sum(len(m) for m in overlay["entry_set"].values())
    stubs = (f"+{len(added)}" + (f" ({', '.join(added)})" if added else "")
             + f" -{len(removed)}" + (f" ({', '.join(removed)})" if removed else ""))
    summary = (f"{source}: {n_edits} entry edit(s), {len(overlay['entry_remove'])} removal set(s), "
               f"{len(overlay['city'])} city set(s), {len(overlay['join'])} join(s) [stubs {stubs}]")
    if dry_run:
        click.echo(json.dumps(overlay, indent=2, sort_keys=True))
        click.echo(f"# {summary} (dry run)", err=True)
        return

    OVERRIDES.write_text(json.dumps(overlay, indent=1, sort_keys=True) + "\n")
    SEEDS.write_text(json.dumps(seeds, indent=2, ensure_ascii=False))
    ROSTER.write_text(json.dumps(roster, indent=1) + "\n")
    click.echo(f"wrote {OVERRIDES.name} (+ seeds/roster sync): {summary}")


if __name__ == "__main__":
    main()
