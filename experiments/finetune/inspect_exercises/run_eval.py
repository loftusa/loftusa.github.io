#!/usr/bin/env python
"""
Runner script for Inspect exercises with local PEFT model.

Usage:
    uv run python run_eval.py ex1         # Run exercise 1 with base model
    uv run python run_eval.py ex1 --dpo   # Run exercise 1 with DPO model
    uv run python run_eval.py ex1 --limit 5  # Limit samples
"""

import sys
import argparse

# Register the PEFT model API first
from model_api import register_peft_model, MODEL_NAME, CHECKPOINT_PATH
register_peft_model()

from inspect_ai import eval as inspect_eval


def main():
    parser = argparse.ArgumentParser(description="Run Inspect exercises with local model")
    parser.add_argument("exercise", choices=["ex1", "ex2", "ex3", "ex4", "ex5", "ex6"],
                        help="Which exercise to run")
    parser.add_argument("--dpo", action="store_true",
                        help="Use DPO checkpoint instead of base model")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of samples")
    parser.add_argument("--checkpoint", type=str, default=CHECKPOINT_PATH,
                        help="Path to checkpoint (default: checkpoint-126)")
    args = parser.parse_args()

    # Import the task
    # Use solutions for tasks that aren't implemented yet
    if args.exercise == "ex1":
        from ex1_hello_world import hello_world as task_fn
    elif args.exercise == "ex2":
        # Use solution since ex2 is a template
        from solutions import full_dataset as task_fn
    elif args.exercise == "ex3":
        # Use solution since ex3 is a template
        from solutions import refusal_eval as task_fn
    elif args.exercise == "ex4":
        # Use solution since ex4 is a template
        from solutions import judge_eval as task_fn
    elif args.exercise == "ex5":
        # Use solution since ex5 is a template
        from solutions import context_eval as task_fn
    elif args.exercise == "ex6":
        from ex6_dpo_comparison import run_comparison
        run_comparison(limit=args.limit or 20)
        return

    # Build model args
    model_args = {}
    if args.dpo:
        model_args["checkpoint_path"] = args.checkpoint
        print(f"Using DPO model: {args.checkpoint}")
    else:
        print(f"Using base model: {MODEL_NAME}")

    # Run eval
    results = inspect_eval(
        task_fn(),
        model=f"peft/{MODEL_NAME}",
        model_args=model_args,
        limit=args.limit,
    )

    # Print summary
    if results:
        result = results[0]
        print("\n" + "=" * 60)
        print("Results")
        print("=" * 60)
        for score in result.results.scores:
            print(f"Scorer: {score.name}")
            for metric_name, metric in score.metrics.items():
                print(f"  {metric_name}: {metric.value:.3f}")


if __name__ == "__main__":
    main()
