# /// script
# requires-python = ">=3.10"
# dependencies = ["click", "rich"]
# ///
"""Review and fill in affiliations.json for the /coauthorship/ affiliation graph.

The web sweep pre-fills entries (each with a source URL) but leaves them verified=false;
this tool walks you through each person so you can confirm, fix, or add entries fast.
Stub people (no entries found) come first. Progress saves after every person — quit anytime.

    cd experiments/coauthorship && uv run fill_affiliations.py            # review loop
    cd experiments/coauthorship && uv run fill_affiliations.py --stats    # coverage table
    cd experiments/coauthorship && uv run fill_affiliations.py --person "ted kyi"
    cd experiments/coauthorship && uv run fill_affiliations.py --all      # include already-reviewed
"""
import json
import webbrowser
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

HERE = Path(__file__).resolve().parent
DATA = HERE / "affiliations.json"
TYPES = ["lab", "university", "company", "community", "program"]

console = Console()


def load() -> dict:
    assert DATA.exists(), f"{DATA} not found — run the sweep first"
    people = json.loads(DATA.read_text())
    assert isinstance(people, dict) and people, "affiliations.json: expected a non-empty name->record mapping"
    for name, rec in people.items():
        assert isinstance(rec.get("entries"), list), f"{name}: missing entries list"
    return people


def save(people: dict) -> None:
    tmp = DATA.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(people, indent=1, ensure_ascii=False) + "\n")
    tmp.replace(DATA)


def show_person(name: str, rec: dict) -> None:
    table = Table(title=f"{name}" + (f"  ·  {rec['city']}" if rec.get("city") else ""), title_justify="left")
    for col in ("#", "org", "type", "role", "years", "ok", "source"):
        table.add_column(col)
    for i, e in enumerate(rec["entries"]):
        table.add_row(
            str(i), e["org"], e["type"], e.get("role", ""), e.get("years", ""),
            "[green]✓[/]" if e.get("verified") else "[yellow]?[/]",
            (e.get("source") or "")[:60],
        )
    console.print(table)
    if rec.get("notes"):
        console.print(f"  [dim]notes: {rec['notes']}[/]")
    if not rec.get("identity_confident", True):
        console.print("  [red]sweep was NOT confident this is the right person — double-check[/]")


def edit_entry(entry: dict | None) -> dict:
    e = entry or {}
    org = click.prompt("  org", default=e.get("org", ""))
    typ = click.prompt("  type", default=e.get("type", "company"), type=click.Choice(TYPES))
    role = click.prompt("  role", default=e.get("role", ""))
    years = click.prompt("  years (e.g. 2023– or 2019–2022)", default=e.get("years", ""))
    source = click.prompt("  source url", default=e.get("source", ""))
    assert org.strip(), "org is required"
    return {**e, "org": org.strip(), "type": typ, "role": role.strip(),
            "years": years.strip(), "source": source.strip(), "verified": True}


def review_person(name: str, rec: dict) -> str:
    """Interactive loop for one person. Returns 'next' or 'quit'."""
    while True:
        console.print()
        show_person(name, rec)
        cmd = click.prompt(
            "[a]ccept all  [e N]dit  [d N]elete  [n]ew  [c]ity  [o N]pen source  [s]kip  [q]uit",
            default="a", show_default=False,
        ).strip().lower()
        op, _, arg = cmd.partition(" ")
        try:
            if op == "a":
                for e in rec["entries"]:
                    e["verified"] = True
                rec["reviewed"] = True
                return "next"
            elif op == "e":
                i = int(arg)
                rec["entries"][i] = edit_entry(rec["entries"][i])
            elif op == "d":
                dropped = rec["entries"].pop(int(arg))
                console.print(f"  dropped {dropped['org']}")
            elif op == "n":
                rec["entries"].append(edit_entry(None))
            elif op == "c":
                rec["city"] = click.prompt("  city", default=rec.get("city", "")).strip()
            elif op == "o":
                url = rec["entries"][int(arg)].get("source")
                webbrowser.open(url) if url else console.print("  [red]no source url[/]")
            elif op == "s":
                return "next"
            elif op == "q":
                return "quit"
            else:
                console.print("  [red]unknown command[/]")
        except (ValueError, IndexError):
            console.print("  [red]bad entry number[/]")


def print_stats(people: dict) -> None:
    table = Table(title="affiliations.json coverage")
    for col in ("person", "entries", "verified", "reviewed", "city"):
        table.add_column(col)
    for name in sorted(people, key=lambda n: (len(people[n]["entries"]) > 0, people[n].get("reviewed", False))):
        rec = people[name]
        n, v = len(rec["entries"]), sum(e.get("verified", False) for e in rec["entries"])
        style = "red" if n == 0 else ("green" if rec.get("reviewed") else "yellow")
        table.add_row(f"[{style}]{name}[/]", str(n), f"{v}/{n}", "✓" if rec.get("reviewed") else "", rec.get("city", ""))
    console.print(table)
    stubs = sum(1 for r in people.values() if not r["entries"])
    done = sum(1 for r in people.values() if r.get("reviewed"))
    console.print(f"{len(people)} people · {done} reviewed · {stubs} stubs")


@click.command()
@click.option("--stats", is_flag=True, help="Print coverage table and exit.")
@click.option("--person", default=None, help="Review a single person by name.")
@click.option("--all", "include_reviewed", is_flag=True, help="Also revisit already-reviewed people.")
def main(stats: bool, person: str | None, include_reviewed: bool) -> None:
    people = load()
    if stats:
        print_stats(people)
        return

    if person:
        assert person in people, f"unknown person {person!r} — names are lowercase, e.g. 'ted kyi'"
        queue = [person]
    else:
        # stubs first (most needs you), then unverified, reviewed last/skipped
        queue = [n for n in sorted(people, key=lambda n: len(people[n]["entries"]) > 0)
                 if include_reviewed or not people[n].get("reviewed")]

    console.print(f"[bold]{len(queue)} people to review[/] (saves after each — quit anytime)")
    for i, name in enumerate(queue, 1):
        console.rule(f"{i}/{len(queue)}")
        action = review_person(name, people[name])
        save(people)
        if action == "quit":
            break
    print_stats(people)


if __name__ == "__main__":
    main()
