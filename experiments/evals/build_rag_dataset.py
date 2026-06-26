"""
Build and verify the RAG evaluation dataset.

For each question in rag_eval_dataset.jsonl, verifies that retrieval
returns chunks from the expected sources (where applicable).

Run: uv run experiments/evals/build_rag_dataset.py
     uv run experiments/evals/build_rag_dataset.py --verify
"""

import json
import sys
from collections import Counter
from pathlib import Path

import click

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from experiments.rag_context import load_collection, retrieve_context

EVALS_DIR = Path(__file__).parent
DATASET_PATH = EVALS_DIR / "rag_eval_dataset.jsonl"


def load_dataset() -> list[dict]:
    records = []
    with open(DATASET_PATH) as f:
        for line in f:
            records.append(json.loads(line))
    return records


@click.command()
@click.option("--verify", is_flag=True, help="Verify retrieval returns expected sources.")
@click.option("--n-results", default=5, type=int, help="Number of chunks to retrieve per query.")
@click.option("--max-distance", default=1.3, type=float, help="Max distance threshold.")
def main(verify: bool, n_results: int, max_distance: float):
    records = load_dataset()

    cats = Counter(r["category"] for r in records)
    print(f"RAG eval dataset: {len(records)} samples")
    print("\nBy category:")
    for cat, count in sorted(cats.items(), key=lambda x: -x[1]):
        print(f"  {cat:25s} {count:3d}")

    if not verify:
        return

    collection = load_collection()
    print(f"\nVerifying retrieval ({collection.count()} chunks in index)...\n")

    hits = 0
    misses = 0
    no_expected = 0

    for record in records:
        expected_sources = record.get("expected_sources", [])
        if not expected_sources:
            no_expected += 1
            continue

        chunks = retrieve_context(
            collection, record["prompt"], n_results=n_results, max_distance=max_distance
        )
        retrieved_sources = [c["source"] for c in chunks]

        # Check if any expected source appears in retrieved sources (prefix match)
        found = any(
            any(exp in ret for ret in retrieved_sources) for exp in expected_sources
        )

        status = "HIT" if found else "MISS"
        if found:
            hits += 1
        else:
            misses += 1

        print(f"[{status:4s}] {record['prompt'][:60]}")
        if not found:
            print(f"       Expected: {expected_sources}")
            print(f"       Got:      {retrieved_sources}")

    total = hits + misses
    print(f"\nRetrieval verification: {hits}/{total} hits ({hits/total:.0%})")
    if no_expected:
        print(f"  ({no_expected} samples had no expected_sources — hallucination traps)")


if __name__ == "__main__":
    main()
