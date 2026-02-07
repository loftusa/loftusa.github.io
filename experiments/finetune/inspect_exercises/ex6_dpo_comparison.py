"""
Exercise 6: DPO Model Comparison

Goal: Compare refusal rates between base Qwen and your DPO-finetuned model

You need to:
1. Reuse your code from ex2 (dataset), ex3 (scorer), ex5 (solver)
2. Create a task that combines them
3. Use inspect_eval() to run on both base and DPO models
4. Extract and compare the accuracy metrics

Run: uv run python ex6_dpo_comparison.py

Key API:
    from inspect_ai import eval as inspect_eval

    # Base model
    base_results = inspect_eval(
        my_task(),
        model=f"peft/{MODEL_NAME}",
        limit=20,
    )

    # DPO model
    dpo_results = inspect_eval(
        my_task(),
        model=f"peft/{MODEL_NAME}",
        model_args={"checkpoint_path": CHECKPOINT_PATH},
        limit=20,
    )

    # Extract accuracy
    acc = results[0].results.scores[0].metrics["accuracy"].value

Available from model_api:
    MODEL_NAME = "Qwen/Qwen2-0.5B-Instruct"
    CHECKPOINT_PATH = ".../checkpoint-126"
    SYSTEM_PROMPT, RESUME

Dataset path: Path(__file__).parent.parent / "local_datasets/dpo_eval.jsonl"
"""

from model_api import register_peft_model, MODEL_NAME, CHECKPOINT_PATH, SYSTEM_PROMPT, RESUME
register_peft_model()

# Your code here

if __name__ == "__main__":
    # run_comparison(limit=20)
    pass
