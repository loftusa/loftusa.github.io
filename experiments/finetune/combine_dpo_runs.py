"""
Combine multiple DPO generation runs into a single consensus dataset.

For each prompt that appears in at least `threshold` fraction of runs:
- Include in final dataset
- Use the shortest "chosen" response (prefer concise refusals)
- Use the modal (most common) "rejected" response

Usage:
    uv run experiments/finetune/combine_dpo_runs.py
    uv run experiments/finetune/combine_dpo_runs.py --threshold 0.8
    uv run experiments/finetune/combine_dpo_runs.py --min-runs 7
"""

from collections import Counter, defaultdict
from pathlib import Path
import json
import click
from rich.console import Console
from rich.table import Table


def load_all_runs(logs_dir: Path) -> tuple[dict[str, list[dict]], int]:
    """
    Load all dpo_pairs_*.jsonl files.

    Returns:
        prompt_data: {prompt: [{"run_id": str, "chosen": str, "rejected": str}, ...]}
        num_runs: total number of run files found
    """
    prompt_data: dict[str, list[dict]] = defaultdict(list)
    files = sorted(logs_dir.glob("dpo_pairs_*.jsonl"))

    if not files:
        raise FileNotFoundError(f"No dpo_pairs_*.jsonl files found in {logs_dir}")

    for f in files:
        run_id = f.stem.replace("dpo_pairs", "").lstrip("_") or "0"
        with f.open() as fp:
            for line in fp:
                line = line.strip()
                if line:
                    d = json.loads(line)
                    prompt_data[d["prompt"]].append({
                        "run_id": run_id,
                        "chosen": d["chosen"],
                        "rejected": d["rejected"]
                    })

    return dict(prompt_data), len(files)


def get_shortest_chosen(entries: list[dict]) -> str:
    """Return the shortest 'chosen' response from a list of entries."""
    return min((e["chosen"] for e in entries), key=len)


def get_modal_rejected(entries: list[dict]) -> str:
    """Return the most common 'rejected' response from a list of entries."""
    counter = Counter(e["rejected"] for e in entries)
    return counter.most_common(1)[0][0]


@click.command()
@click.option(
    "--logs-dir",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Directory containing dpo_pairs_*.jsonl files. Defaults to experiments/logs/",
)
@click.option(
    "--output",
    type=click.Path(path_type=Path),
    default=None,
    help="Output file path. Defaults to dpo_combined.jsonl in logs dir",
)
@click.option(
    "--threshold",
    type=float,
    default=0.5,
    help="Minimum fraction of runs a prompt must appear in (0.0-1.0). Default: 0.5",
)
@click.option(
    "--min-runs",
    type=int,
    default=None,
    help="Minimum number of runs a prompt must appear in. Overrides --threshold if set.",
)
@click.option(
    "--verbose", "-v",
    is_flag=True,
    help="Print detailed information about each prompt",
)
def main(
    logs_dir: Path | None,
    output: Path | None,
    threshold: float,
    min_runs: int | None,
    verbose: bool,
):
    """Combine multiple DPO runs into a consensus dataset based on agreement."""
    console = Console()

    # Setup paths
    if logs_dir is None:
        logs_dir = Path(__file__).parent.parent / "logs"

    if output is None:
        output = logs_dir / "dpo_combined.jsonl"

    # Load data
    console.print(f"[bold]Loading DPO runs from {logs_dir}...[/bold]")
    prompt_data, num_runs = load_all_runs(logs_dir)
    console.print(f"Found {len(prompt_data)} unique prompts across {num_runs} runs")

    # Determine threshold
    if min_runs is not None:
        actual_min_runs = min_runs
    else:
        actual_min_runs = max(1, int(threshold * num_runs))

    console.print(f"Threshold: {actual_min_runs}/{num_runs} runs ({actual_min_runs/num_runs*100:.0f}%)")

    # Process prompts
    included = []
    excluded = []

    for prompt, entries in prompt_data.items():
        n = len(entries)
        if n >= actual_min_runs:
            chosen = get_shortest_chosen(entries)
            rejected = get_modal_rejected(entries)
            included.append({
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rejected,
                "agreement": n,
                "agreement_pct": n / num_runs * 100,
            })
        else:
            excluded.append({
                "prompt": prompt,
                "agreement": n,
                "agreement_pct": n / num_runs * 100,
            })

    # Sort by agreement (highest first)
    included.sort(key=lambda x: x["agreement"], reverse=True)
    excluded.sort(key=lambda x: x["agreement"], reverse=True)

    # Write output
    with output.open("w") as f:
        for item in included:
            dpo_pair = {
                "prompt": item["prompt"],
                "chosen": item["chosen"],
                "rejected": item["rejected"],
            }
            f.write(json.dumps(dpo_pair) + "\n")

    # Summary statistics
    console.print()
    console.print("[bold green]═══ SUMMARY ═══[/bold green]")
    console.print(f"Total unique prompts: {len(prompt_data)}")
    console.print(f"Included (≥{actual_min_runs} runs): {len(included)}")
    console.print(f"Excluded (<{actual_min_runs} runs): {len(excluded)}")
    console.print(f"Output written to: {output}")

    # Agreement distribution
    console.print()
    console.print("[bold]Agreement distribution:[/bold]")
    agreement_counts = Counter(len(entries) for entries in prompt_data.values())

    table = Table(show_header=True, header_style="bold")
    table.add_column("Runs", justify="right")
    table.add_column("# Prompts", justify="right")
    table.add_column("Status", justify="center")

    for n_runs in range(1, num_runs + 1):
        count = agreement_counts.get(n_runs, 0)
        status = "✓ included" if n_runs >= actual_min_runs else "✗ excluded"
        table.add_row(
            f"{n_runs}/{num_runs}",
            str(count),
            status,
        )

    console.print(table)

    # Verbose output
    if verbose:
        console.print()
        console.print("[bold]Included prompts:[/bold]")
        for item in included[:10]:
            console.print(f"  [{item['agreement']}/{num_runs}] {item['prompt'][:60]}...")
            console.print(f"    → chosen: {item['chosen'][:60]}...")

        if len(included) > 10:
            console.print(f"  ... and {len(included) - 10} more")

        if excluded:
            console.print()
            console.print("[bold]Excluded prompts (insufficient agreement):[/bold]")
            for item in excluded[:5]:
                console.print(f"  [{item['agreement']}/{num_runs}] {item['prompt'][:60]}...")

            if len(excluded) > 5:
                console.print(f"  ... and {len(excluded) - 5} more")


if __name__ == "__main__":
    main()
