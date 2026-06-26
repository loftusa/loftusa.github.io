"""
Retrieval quality analysis for the RAG eval dataset.

For each question, runs retrieval and measures:
- Whether expected sources appear in top-K results
- Recall@K and MRR (mean reciprocal rank)
- Which questions have zero relevant retrieval

No LLM calls needed — fast and cheap.

Run: uv run experiments/evals/analyze_retrieval.py
     uv run experiments/evals/analyze_retrieval.py --n-results 10
     uv run experiments/evals/analyze_retrieval.py --verbose
"""

import json
import sys
from collections import defaultdict
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


def source_matches(expected: str, retrieved: str) -> bool:
    """Check if an expected source matches a retrieved source (prefix match)."""
    return expected in retrieved


@click.command()
@click.option("--n-results", default=5, type=int, help="Number of chunks to retrieve.")
@click.option("--max-distance", default=1.5, type=float, help="Max distance threshold.")
@click.option("--verbose", is_flag=True, help="Show per-question details.")
def main(n_results: int, max_distance: float, verbose: bool):
    records = load_dataset()
    collection = load_collection()
    print(f"Index: {collection.count()} chunks")
    print(f"Dataset: {len(records)} questions")
    print(f"Settings: n_results={n_results}, max_distance={max_distance}\n")

    category_stats = defaultdict(lambda: {"total": 0, "hits": 0, "rr_sum": 0.0})
    zero_retrieval = []
    no_expected = 0

    for record in records:
        expected_sources = record.get("expected_sources", [])
        category = record["category"]

        if not expected_sources:
            no_expected += 1
            # For hallucination traps: check if retrieval returns nothing relevant
            chunks = retrieve_context(
                collection, record["prompt"], n_results=n_results, max_distance=max_distance
            )
            if verbose:
                status = f"{len(chunks)} chunks" if chunks else "no chunks"
                print(f"[TRAP ] {record['prompt'][:60]} → {status}")
            continue

        chunks = retrieve_context(
            collection, record["prompt"], n_results=n_results, max_distance=max_distance
        )
        retrieved_sources = [c["source"] for c in chunks]

        category_stats[category]["total"] += 1

        # Find reciprocal rank: position of first relevant source
        rr = 0.0
        found = False
        for rank, ret_src in enumerate(retrieved_sources, 1):
            if any(source_matches(exp, ret_src) for exp in expected_sources):
                rr = 1.0 / rank
                found = True
                break

        if found:
            category_stats[category]["hits"] += 1
            category_stats[category]["rr_sum"] += rr
        else:
            zero_retrieval.append(record["prompt"])

        if verbose:
            status = f"HIT (RR={rr:.2f})" if found else "MISS"
            print(f"[{status:12s}] {record['prompt'][:60]}")
            if not found:
                print(f"  Expected: {expected_sources}")
                print(f"  Got:      {retrieved_sources}")

    # Print summary
    print(f"\n{'=' * 60}")
    print(f"  Retrieval Quality Analysis (top-{n_results})")
    print(f"{'=' * 60}")
    header = f"{'Category':<28}| {'Queries':>7} | {'Recall':>7} | {'MRR':>6}"
    sep = "-" * 28 + "+" + "-" * 9 + "+" + "-" * 9 + "+" + "-" * 8
    print(header)
    print(sep)

    total_queries = 0
    total_hits = 0
    total_rr = 0.0
    for cat, stats in sorted(category_stats.items()):
        total = stats["total"]
        hits = stats["hits"]
        recall = hits / total if total else 0
        mrr = stats["rr_sum"] / total if total else 0
        print(f"{cat:<28}| {total:>7} | {recall:>6.1%} | {mrr:>5.3f}")
        total_queries += total
        total_hits += hits
        total_rr += stats["rr_sum"]

    print(sep)
    overall_recall = total_hits / total_queries if total_queries else 0
    overall_mrr = total_rr / total_queries if total_queries else 0
    print(f"{'OVERALL':<28}| {total_queries:>7} | {overall_recall:>6.1%} | {overall_mrr:>5.3f}")

    if no_expected:
        print(f"\n  ({no_expected} hallucination trap questions — no expected sources)")

    if zero_retrieval:
        print(f"\n  Zero-retrieval questions ({len(zero_retrieval)}):")
        for q in zero_retrieval:
            print(f"    - {q[:70]}")


if __name__ == "__main__":
    main()
