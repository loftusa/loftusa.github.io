# /// script
# requires-python = ">=3.10"
# dependencies = ["click", "httpx"]
# ///
"""Pull the merged crowd-correction overlay from the Fly API into the git-tracked overrides.json.

The website POSTs one append-only *event* per edit to the Fly API. The API's open
`GET /coauthorship/overlay` endpoint returns those events already folded into the durable
`overrides.json` contract (same `fold_events` in `overrides.py`), so the default mode needs **no
token**. Run nightly, before `build_graph.py`:

    cd experiments/coauthorship
    uv run merge_corrections.py                       # GET the open overlay, rewrite overrides.json
    uv run merge_corrections.py --raw                 # fold the Bearer-protected raw log instead
    uv run merge_corrections.py --from events.jsonl   # offline: fold a local JSONL
    uv run merge_corrections.py --dry-run             # print the merged overrides, write nothing

Raw event shape (one JSON object per line)::

    {"type": "node_label", "payload": {"id": "...", "label": "..."}, "editor": "?", "note": "?", "ts": "ISO8601"}

`fold_events` (in overrides.py) is pure: last-write-wins per node/paper key, ts-aware undo so a
later remove cancels an earlier crowd-add. Idempotent: same log -> same file. To durably revert a
bad edit, DELETE its event from the API log (see chat_api.py::delete_correction) — editing
overrides.json by hand is re-clobbered by the next merge run.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import click
import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent))
from overrides import EMPTY, fold_events  # noqa: E402  (pure folding logic lives in overrides.py)

HERE = Path(__file__).resolve().parent
OVERRIDES = HERE / "overrides.json"
DEFAULT_API = os.environ.get("COAUTHOR_API", "https://llm-resume-restless-thunder-9259.fly.dev")


def parse_jsonl(text: str) -> list[dict]:
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def fetch(url: str, headers: dict) -> httpx.Response | None:
    """GET with the nightly-build fail-safe: any failure (API not deployed, network blip, bad
    token) warns and returns None so the caller leaves overrides.json untouched and exits 0 —
    a correction-fetch hiccup must never abort the graph rebuild that follows."""
    try:
        r = httpx.get(url, headers=headers, timeout=30)
        r.raise_for_status()
        return r
    except httpx.HTTPError as e:
        click.echo(f"warning: could not fetch corrections ({e}); leaving {OVERRIDES.name} as-is", err=True)
        return None


@click.command()
@click.option("--api", default=DEFAULT_API, help="Fly API base URL.")
@click.option("--raw", is_flag=True,
              help="Fold the Bearer-protected raw event log instead of the open overlay (audit path).")
@click.option("--token", envvar="COAUTHOR_TOKEN", default=None, help="Bearer token; only used with --raw.")
@click.option("--from", "from_file", type=click.Path(exists=True), default=None,
              help="Fold a local JSONL file instead of fetching from the API (offline/testing).")
@click.option("--dry-run", is_flag=True, help="Print the merged overrides; do not write the file.")
def main(api: str, raw: bool, token: str | None, from_file: str | None, dry_run: bool) -> None:
    base = api.rstrip("/")
    source = ""
    if from_file:
        events = parse_jsonl(Path(from_file).read_text())
        merged, source = fold_events(events), f"{len(events)} events ({from_file})"
    elif raw:
        r = fetch(f"{base}/coauthorship/corrections",
                  {"Authorization": f"Bearer {token}"} if token else {})
        if r is None:
            return
        events = parse_jsonl(r.text)
        merged, source = fold_events(events), f"{len(events)} raw events"
    else:
        # default: the open overlay endpoint already returns fold_events(log) — no secret needed
        r = fetch(f"{base}/coauthorship/overlay", {})
        if r is None:
            return
        merged, source = {**EMPTY, **r.json()}, "live overlay"

    n = sum(len(merged[k]) for k in ("remove_nodes", "remove_papers", "add_papers", "remove_edges")) \
        + sum(len(merged[k]) for k in ("node_label", "node_community", "node_url", "node_photo", "paper_rename"))
    blob = json.dumps(merged, indent=2)
    if dry_run:
        click.echo(blob)
        click.echo(f"# {source} -> {n} override(s) (dry run, not written)", err=True)
        return
    OVERRIDES.write_text(blob + "\n")
    click.echo(f"wrote {OVERRIDES.name}: {source} -> {n} override(s)")


if __name__ == "__main__":
    main()
