"""
Exercise 4: Model-Graded Scorer (LLM-as-Judge)

Goal: Write a scorer that uses an LLM to judge model output vs reference

You need to:
1. Use get_model() to get a grader model (e.g., "anthropic/claude-sonnet-4-20250514")
2. Build a judge prompt comparing model output vs target
3. Call await grader.generate(prompt) and parse A/B/TIE response
4. Return Score based on who wins

Run: uv run python run_eval.py ex4 --limit 5

Imports you'll need:
    from inspect_ai import Task, task
    from inspect_ai.dataset import Sample, json_dataset
    from inspect_ai.solver import generate, TaskState
    from inspect_ai.scorer import scorer, Score, Target, accuracy
    from inspect_ai.model import get_model

Key APIs:
    grader = get_model("anthropic/claude-sonnet-4-20250514")
    result = await grader.generate(prompt)
    judgment = result.completion  # The model's text response

    state.input_text       # Original prompt
    state.output.completion # Model's response
    target.text            # Reference response

Dataset path: Path(__file__).parent.parent / "local_datasets/dpo_eval.jsonl"
"""

from model_api import register_peft_model
register_peft_model()

# Your code here
