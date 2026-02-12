"""CLI viewer for jailbreak_dataset.jsonl — explore the eval dataset quickly."""

import json
import random
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from collections import Counter

DATASET_PATH = Path(__file__).parent / "jailbreak_dataset.jsonl"
console = Console()


def load_dataset() -> list[dict]:
    with open(DATASET_PATH) as f:
        return [json.loads(line) for line in f if line.strip()]


@click.group()
def cli():
    """Explore the jailbreak evaluation dataset."""
    pass


@cli.command()
def summary():
    """High-level summary: counts by category and behavior."""
    data = load_dataset()
    cat_counts = Counter(d["category"] for d in data)
    beh_counts = Counter(d["expected_behavior"] for d in data)

    console.print(f"\n[bold]Total samples:[/bold] {len(data)}")
    console.print(f"[bold]Refuse:[/bold] {beh_counts.get('refuse', 0)}  |  [bold]Answer:[/bold] {beh_counts.get('answer', 0)}\n")

    table = Table(title="Samples by Category")
    table.add_column("Category", style="cyan")
    table.add_column("Total", justify="right")
    table.add_column("Refuse", justify="right", style="red")
    table.add_column("Answer", justify="right", style="green")

    cat_beh = {}
    for d in data:
        cat_beh.setdefault(d["category"], Counter())[d["expected_behavior"]] += 1

    for cat in sorted(cat_beh, key=lambda c: -sum(cat_beh[c].values())):
        counts = cat_beh[cat]
        table.add_row(cat, str(sum(counts.values())), str(counts.get("refuse", 0)), str(counts.get("answer", 0)))

    console.print(table)


@cli.command(name="list")
@click.option("-c", "--category", default=None, help="Filter by category")
@click.option("-b", "--behavior", default=None, type=click.Choice(["refuse", "answer"]), help="Filter by expected behavior")
@click.option("-n", "--limit", default=None, type=int, help="Max items to show")
@click.option("--short", is_flag=True, help="Truncate prompts to 80 chars")
def list_items(category, behavior, limit, short):
    """List prompts with optional filtering."""
    data = load_dataset()
    if category:
        data = [d for d in data if d["category"] == category]
    if behavior:
        data = [d for d in data if d["expected_behavior"] == behavior]
    if limit:
        data = data[:limit]

    table = Table(show_lines=True)
    table.add_column("#", justify="right", style="dim", width=4)
    table.add_column("Behavior", width=8)
    table.add_column("Category", style="cyan", width=18)
    table.add_column("Prompt")
    table.add_column("Target", style="dim", max_width=40)

    for i, d in enumerate(data):
        beh_style = "red" if d["expected_behavior"] == "refuse" else "green"
        prompt = d["prompt"][:80] + "…" if short and len(d["prompt"]) > 80 else d["prompt"]
        target = d["target"][:40] + "…" if len(d["target"]) > 40 else d["target"]
        table.add_row(str(i), f"[{beh_style}]{d['expected_behavior']}[/]", d["category"], prompt, target)

    console.print(table)
    console.print(f"\n[dim]{len(data)} items shown[/dim]")


@cli.command()
@click.argument("index", type=int)
def show(index):
    """Show full details for a single sample by index."""
    data = load_dataset()
    if index < 0 or index >= len(data):
        console.print(f"[red]Index {index} out of range (0-{len(data)-1})[/red]")
        return
    d = data[index]
    beh_color = "red" if d["expected_behavior"] == "refuse" else "green"
    console.print(Panel(
        f"[bold]Prompt:[/bold]\n{d['prompt']}\n\n"
        f"[bold]Expected:[/bold] [{beh_color}]{d['expected_behavior']}[/]\n"
        f"[bold]Category:[/bold] [cyan]{d['category']}[/cyan]\n"
        f"[bold]Target:[/bold] {d['target']}",
        title=f"Sample #{index}",
    ))


@cli.command()
@click.option("-n", "--count", default=5, help="Number of random samples")
@click.option("-c", "--category", default=None, help="Filter by category")
@click.option("-b", "--behavior", default=None, type=click.Choice(["refuse", "answer"]), help="Filter by expected behavior")
def sample(count, category, behavior):
    """Show random samples from the dataset."""
    data = load_dataset()
    if category:
        data = [d for d in data if d["category"] == category]
    if behavior:
        data = [d for d in data if d["expected_behavior"] == behavior]
    if not data:
        console.print("[red]No matching samples[/red]")
        return

    picks = random.sample(data, min(count, len(data)))
    for d in picks:
        beh_color = "red" if d["expected_behavior"] == "refuse" else "green"
        console.print(Panel(
            f"{d['prompt']}\n\n"
            f"[{beh_color}]{d['expected_behavior']}[/] · [cyan]{d['category']}[/cyan]\n"
            f"[dim]{d['target']}[/dim]",
        ))


@cli.command()
@click.argument("query")
def search(query):
    """Search prompts and targets for a substring (case-insensitive)."""
    data = load_dataset()
    q = query.lower()
    hits = [d for d in data if q in d["prompt"].lower() or q in d["target"].lower()]

    if not hits:
        console.print(f"[yellow]No matches for '{query}'[/yellow]")
        return

    table = Table(show_lines=True, title=f"Search: '{query}'")
    table.add_column("Behavior", width=8)
    table.add_column("Category", style="cyan", width=18)
    table.add_column("Prompt")
    table.add_column("Target", style="dim", max_width=40)

    for d in hits:
        beh_style = "red" if d["expected_behavior"] == "refuse" else "green"
        table.add_row(f"[{beh_style}]{d['expected_behavior']}[/]", d["category"], d["prompt"], d["target"])

    console.print(table)
    console.print(f"\n[dim]{len(hits)} matches[/dim]")


if __name__ == "__main__":
    cli()
