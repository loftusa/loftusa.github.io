"""
Exercise 3: Custom Scorer - Refusal Detection

Goal: Write a custom scorer that detects refusal patterns via regex

You need to:
1. Define REFUSAL_PATTERNS list (regex patterns)
2. Write a @scorer function that checks if output matches any pattern
3. Return Score(value="C") if refusal detected, "I" otherwise

Run: uv run python run_eval.py ex3 --limit 10

Imports you'll need:
    from inspect_ai import Task, task
    from inspect_ai.dataset import Sample, json_dataset
    from inspect_ai.solver import generate, TaskState
    from inspect_ai.scorer import scorer, Score, Target, accuracy

Scorer pattern:
    @scorer(metrics=[accuracy()])
    def my_scorer():
        async def score(state: TaskState, target: Target) -> Score:
            output_text = state.output.completion
            return Score(value="C" or "I", answer=output_text[:200])
        return score

Dataset path: Path(__file__).parent.parent / "local_datasets/dpo_eval.jsonl"
"""

from model_api import register_peft_model
register_peft_model()

# Your code here
