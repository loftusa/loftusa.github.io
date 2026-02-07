"""
Exercise 2: Custom Dataset Loading

Goal: Load dpo_eval.jsonl (221 samples) into Inspect

You need to:
1. Use json_dataset() to load the JSONL file
2. Write a record_to_sample() function that maps:
   - prompt -> input
   - chosen -> target
   - rejected -> metadata
3. Create a Task with the dataset

Run: uv run python run_eval.py ex2 --limit 5

Imports you'll need:
    from inspect_ai import Task, task
    from inspect_ai.dataset import Sample, json_dataset
    from inspect_ai.solver import generate
    from inspect_ai.scorer import includes

Dataset path: Path(__file__).parent.parent / "local_datasets/dpo_eval.jsonl"
"""

from model_api import register_peft_model
register_peft_model()

# Your code here
