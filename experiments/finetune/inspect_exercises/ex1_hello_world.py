"""
Exercise 1: Hello World

Goal: Create a basic Inspect task that checks if the model mentions "Alex"

You need to:
1. Create 5 Sample objects with questions about Alex
2. Wrap them in a MemoryDataset
3. Create a Task with generate() solver and includes() scorer

Run: uv run python run_eval.py ex1 --limit 5

Imports you'll need:
    from inspect_ai import Task, task
    from inspect_ai.dataset import MemoryDataset, Sample
    from inspect_ai.solver import generate
    from inspect_ai.scorer import includes
"""

from model_api import register_peft_model
register_peft_model()

# Your code here
