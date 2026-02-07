#!/usr/bin/env python
"""
Runner script for Inspect jailbreak evaluation exercises.

Usage:
    uv run run_eval.py ex1                    # Run exercise 1
    uv run run_eval.py ex5 --limit 10         # Limit samples
    uv run run_eval.py ex6                    # Full eval with per-category breakdown
    uv run run_eval.py solutions:context_eval # Run a specific solution task
"""

import argparse

import model_api  # noqa: F401 — sets up Cerebras env vars

from inspect_ai import eval as inspect_eval

MODEL = "openai/gpt-oss-120b"


def main():
    parser = argparse.ArgumentParser(description="Run Inspect jailbreak eval exercises")
    parser.add_argument(
        "exercise",
        choices=["ex1", "ex2", "ex3", "ex4", "ex5", "ex6"],
        help="Which exercise to run",
    )
    parser.add_argument("--limit", type=int, default=None, help="Limit number of samples")
    parser.add_argument(
        "--use-solutions", action="store_true",
        help="Use reference solutions instead of exercise files",
    )
    args = parser.parse_args()

    # Exercise 6 runs programmatically (not via inspect eval CLI)
    if args.exercise == "ex6":
        if args.use_solutions:
            from solutions import main as run_full
        else:
            print("Exercise 6 is a standalone script. Run directly:")
            print("  uv run ex6_jailbreak_eval.py")
            print("  uv run ex6_jailbreak_eval.py 20  # limit to 20 samples")
            print("\nOr use --use-solutions to run the reference implementation:")
            print("  uv run run_eval.py ex6 --use-solutions")
            return
        run_full(limit=args.limit)
        return

    # Map exercise to task functions
    task_map = {
        "ex1": ("ex1_hello_world", "jailbreak_hello_world"),
        "ex2": ("ex2_jailbreak_dataset", "jailbreak_dataset"),
        "ex3": ("ex3_refusal_scorer", "refusal_eval"),
        "ex4": ("ex4_llm_judge", "judge_eval"),
        "ex5": ("ex5_system_context", "context_eval"),
    }

    solution_map = {
        "ex1": "jailbreak_hello_world",
        "ex2": "jailbreak_dataset",
        "ex3": "refusal_eval",
        "ex4": "judge_eval",
        "ex5": "context_eval",
    }

    if args.use_solutions:
        import solutions
        task_fn = getattr(solutions, solution_map[args.exercise])
    else:
        module_name, func_name = task_map[args.exercise]
        module = __import__(module_name)
        task_fn = getattr(module, func_name)

    print(f"Running {args.exercise} with model {MODEL}")

    results = inspect_eval(
        task_fn(),
        model=MODEL,
        limit=args.limit,
    )

    # Print summary
    if results:
        result = results[0]
        print("\n" + "=" * 60)
        print("Results")
        print("=" * 60)
        for s in result.results.scores:
            print(f"Scorer: {s.name}")
            for metric_name, metric in s.metrics.items():
                print(f"  {metric_name}: {metric.value:.3f}")


if __name__ == "__main__":
    main()
