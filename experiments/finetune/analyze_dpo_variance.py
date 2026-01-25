"""
Analyze variance across multiple DPO data generation runs.

Answers:
1. How consistently does the classifier identify the same prompts as needing DPO pairs?
2. How similar are the generated refusals for the same prompt across runs?

Usage:
    uv run experiments/finetune/analyze_dpo_variance.py
"""

from collections import defaultdict
import json
from pathlib import Path
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


def load_all_runs(logs_dir: Path) -> dict[str, list[dict]]:
    """
    Load all dpo_pairs_*.jsonl files.

    Returns: {prompt: [{"run_id": str, "chosen": str, "rejected": str}, ...]}
    """
    prompt_data: dict[str, list[dict]] = defaultdict(list)
    files = sorted(logs_dir.glob("dpo_pairs_*.jsonl"))

    if not files:
        raise FileNotFoundError(f"No dpo_pairs_*.jsonl files found in {logs_dir}")

    print(f"Found {len(files)} DPO result files")

    for f in files:
        run_id = f.stem.replace("dpo_pairs", "").lstrip("_") or "0"
        lines = f.read_text().strip().split("\n")
        for line in lines:
            if line:
                d = json.loads(line)
                prompt_data[d["prompt"]].append({
                    "run_id": run_id,
                    "chosen": d["chosen"],
                    "rejected": d["rejected"]
                })

    return dict(prompt_data)


def compute_refusal_similarity(refusals: list[str], model: SentenceTransformer) -> float:
    """Compute mean pairwise cosine similarity of refusal embeddings."""
    if len(refusals) < 2:
        return 1.0  # Only one refusal, perfect agreement with itself

    embeddings = model.encode(refusals)
    sim_matrix = cosine_similarity(embeddings)

    # Get upper triangle (excluding diagonal)
    triu_indices = np.triu_indices(len(refusals), k=1)
    pairwise_sims = sim_matrix[triu_indices]

    return float(np.mean(pairwise_sims))


def analyze_agreement(prompt_data: dict[str, list[dict]], num_runs: int) -> pd.DataFrame:
    """
    Analyze agreement across runs.

    Returns DataFrame with columns: prompt, num_runs, agreement_pct, refusals
    """
    records = []
    for prompt, entries in prompt_data.items():
        records.append({
            "prompt": prompt,
            "num_runs": len(entries),
            "agreement_pct": len(entries) / num_runs * 100,
            "refusals": [e["chosen"] for e in entries],
            "run_ids": [e["run_id"] for e in entries],
        })

    df = pd.DataFrame(records)
    df = df.sort_values("num_runs", ascending=False)
    return df


def create_heatmap_data(prompt_data: dict[str, list[dict]], num_runs: int) -> pd.DataFrame:
    """Create binary matrix: prompts × runs."""
    # Get all run IDs
    all_run_ids = set()
    for entries in prompt_data.values():
        for e in entries:
            all_run_ids.add(e["run_id"])
    run_ids = sorted(all_run_ids, key=lambda x: int(x) if x.isdigit() else 0)

    # Build matrix
    matrix_data = []
    prompts = []
    for prompt, entries in prompt_data.items():
        runs_with_prompt = {e["run_id"] for e in entries}
        row = [1 if rid in runs_with_prompt else 0 for rid in run_ids]
        matrix_data.append(row)
        # Truncate prompt for display
        prompts.append(prompt[:50] + "..." if len(prompt) > 50 else prompt)

    df = pd.DataFrame(matrix_data, index=prompts, columns=[f"Run {r}" for r in run_ids])
    return df


def main():
    # Setup
    logs_dir = Path(__file__).parent.parent / "logs"
    output_dir = Path(__file__).parent / "variance_analysis"
    output_dir.mkdir(exist_ok=True)

    print("Loading sentence embedding model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Load data
    print(f"Loading DPO results from {logs_dir}...")
    prompt_data = load_all_runs(logs_dir)

    # Determine number of runs
    all_run_ids = set()
    for entries in prompt_data.values():
        for e in entries:
            all_run_ids.add(e["run_id"])
    num_runs = len(all_run_ids)

    print(f"Loaded {len(prompt_data)} unique prompts across {num_runs} runs")

    # Analyze agreement
    df = analyze_agreement(prompt_data, num_runs)

    # Compute semantic similarity for prompts with 2+ runs
    print("Computing refusal similarities...")
    similarities = []
    for _, row in df.iterrows():
        if row["num_runs"] >= 2:
            sim = compute_refusal_similarity(row["refusals"], model)
            similarities.append(sim)
        else:
            similarities.append(np.nan)
    df["refusal_similarity"] = similarities

    # =========== SUMMARY STATS ===========
    print("\n" + "=" * 60)
    print("AGREEMENT SUMMARY")
    print("=" * 60)

    perfect_agreement = (df["num_runs"] == num_runs).sum()
    single_run = (df["num_runs"] == 1).sum()
    partial = len(df) - perfect_agreement - single_run

    print(f"Total unique prompts: {len(df)}")
    print(f"Perfect agreement (in all {num_runs} runs): {perfect_agreement} ({perfect_agreement/len(df)*100:.1f}%)")
    print(f"Partial agreement (2-{num_runs-1} runs): {partial} ({partial/len(df)*100:.1f}%)")
    print(f"Single run only: {single_run} ({single_run/len(df)*100:.1f}%)")

    print("\n" + "=" * 60)
    print("REFUSAL SIMILARITY (for prompts in 2+ runs)")
    print("=" * 60)
    valid_sims = df["refusal_similarity"].dropna()
    if len(valid_sims) > 0:
        print(f"Mean similarity: {valid_sims.mean():.3f}")
        print(f"Std similarity: {valid_sims.std():.3f}")
        print(f"Min similarity: {valid_sims.min():.3f}")
        print(f"Max similarity: {valid_sims.max():.3f}")

    # =========== VISUALIZATIONS ===========
    sns.set_theme(style="whitegrid")

    # 1. Histogram: runs per prompt
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(df["num_runs"], bins=range(1, num_runs + 2), edgecolor="black", align="left")
    ax.set_xlabel("Number of runs prompt appears in")
    ax.set_ylabel("Number of prompts")
    ax.set_title("Distribution of Agreement Levels")
    ax.set_xticks(range(1, num_runs + 1))
    plt.tight_layout()
    plt.savefig(output_dir / "agreement_histogram.png", dpi=150)
    print(f"\nSaved: {output_dir / 'agreement_histogram.png'}")

    # 2. Heatmap: prompts × runs (only show prompts with variance)
    df_partial = df[df["num_runs"] < num_runs].head(30)  # Top 30 with variance
    if len(df_partial) > 0:
        # Build heatmap data for these prompts
        partial_prompts = set(df_partial["prompt"])
        partial_data = {p: v for p, v in prompt_data.items() if p in partial_prompts}
        heatmap_df = create_heatmap_data(partial_data, num_runs)

        fig, ax = plt.subplots(figsize=(12, max(8, len(heatmap_df) * 0.3)))
        sns.heatmap(heatmap_df, cmap="YlGnBu", cbar_kws={"label": "Generated DPO pair"}, ax=ax)
        ax.set_title("Prompts with Variance Across Runs")
        plt.tight_layout()
        plt.savefig(output_dir / "variance_heatmap.png", dpi=150)
        print(f"Saved: {output_dir / 'variance_heatmap.png'}")

    # 3. Box plot: refusal similarity distribution
    if len(valid_sims) > 0:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.boxplot(valid_sims, vert=True)
        ax.set_ylabel("Cosine Similarity")
        ax.set_title("Distribution of Refusal Semantic Similarity\n(for prompts appearing in 2+ runs)")
        ax.set_ylim(0, 1)
        plt.tight_layout()
        plt.savefig(output_dir / "refusal_similarity_boxplot.png", dpi=150)
        print(f"Saved: {output_dir / 'refusal_similarity_boxplot.png'}")

    # 4. Save detailed results
    results_df = df[["prompt", "num_runs", "agreement_pct", "refusal_similarity", "run_ids"]].copy()
    results_df.to_csv(output_dir / "agreement_details.csv", index=False)
    print(f"Saved: {output_dir / 'agreement_details.csv'}")

    # 5. Print prompts with most disagreement
    print("\n" + "=" * 60)
    print("PROMPTS WITH MOST DISAGREEMENT (appearing in fewest runs)")
    print("=" * 60)
    low_agreement = df[df["num_runs"] <= 3].head(10)
    for _, row in low_agreement.iterrows():
        print(f"\n[{row['num_runs']}/{num_runs} runs] {row['prompt'][:80]}...")
        if row["num_runs"] >= 2:
            print(f"  Refusal similarity: {row['refusal_similarity']:.3f}")

    # 6. Print prompts with lowest refusal similarity
    print("\n" + "=" * 60)
    print("PROMPTS WITH LOWEST REFUSAL SIMILARITY")
    print("=" * 60)
    low_sim = df[df["refusal_similarity"].notna()].nsmallest(5, "refusal_similarity")
    for _, row in low_sim.iterrows():
        print(f"\n[sim={row['refusal_similarity']:.3f}] {row['prompt'][:80]}...")
        print("  Refusals:")
        for r in row["refusals"][:3]:  # Show first 3
            print(f"    - {r}")


if __name__ == "__main__":
    main()
