"""
Pretty-print a DPO dataset to the terminal.

Usage:
    uv run experiments/finetune/view_dpo.py
    uv run experiments/finetune/view_dpo.py path/to/dpo_file.jsonl
"""

from pathlib import Path
import json
import click
from rich.console import Console
from rich.panel import Panel
from rich.text import Text


@click.command()
@click.argument(
    "file",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    required=False,
)
def main(file: Path | None):
    """Pretty-print a DPO dataset."""
    console = Console()

    if file is None:
        file = Path(__file__).parent.parent / "logs" / "dpo_combined.jsonl"

    if not file.exists():
        console.print(f"[red]File not found: {file}[/red]")
        return

    console.print(f"[bold]Reading: {file}[/bold]\n")

    with file.open() as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            d = json.loads(line)

            prompt_text = Text(d["prompt"], style="bold white")
            chosen_text = Text(d["chosen"], style="green")
            rejected_text = Text(d["rejected"], style="red")

            content = Text()
            content.append("PROMPT\n", style="bold dim")
            content.append(prompt_text)
            content.append("\n\n")
            content.append("CHOSEN ", style="bold green")
            content.append(f"({len(d['chosen'])} chars)\n", style="dim")
            content.append(chosen_text)
            content.append("\n\n")
            content.append("REJECTED ", style="bold red")
            content.append(f"({len(d['rejected'])} chars)\n", style="dim")
            content.append(rejected_text)

            panel = Panel(content, title=f"[bold]#{i}[/bold]", border_style="blue")
            console.print(panel)
            console.print()

    console.print(f"[bold]Total: {i} examples[/bold]")


if __name__ == "__main__":
    main()
