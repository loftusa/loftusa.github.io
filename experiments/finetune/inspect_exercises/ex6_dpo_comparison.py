"""
Exercise 6: DPO Model Comparison (The Payoff!)

CONCEPT: Evaluate your actual DPO-trained model vs base model

YOUR GOAL:
1. Use the PeftModelAPI from model_api.py (already implemented!)
2. Run the same eval on base Qwen and DPO-finetuned Qwen
3. Compare refusal rates between the two

THIS IS THE PAYOFF: By the end, you'll have a proper evaluation comparing
your DPO training to the base model - no more manual testing in dpo_chat.py!

RUN WHEN COMPLETE:
    uv run python ex6_dpo_comparison.py

SUCCESS CRITERIA:
- See refusal rate comparison between base and DPO model
- DPO model should have higher refusal rate on off-topic requests

WHAT YOU'RE LEARNING: Programmatic eval, A/B comparison, results extraction
"""

import re
from pathlib import Path

from model_api import (
    register_peft_model,
    MODEL_NAME,
    CHECKPOINT_PATH,
    SYSTEM_PROMPT,
    RESUME,
)
register_peft_model()

from inspect_ai import Task, task, eval as inspect_eval
from inspect_ai.dataset import json_dataset, Sample
from inspect_ai.solver import Generate, TaskState, solver
from inspect_ai.scorer import scorer, Score, Target, accuracy
from inspect_ai.model import ChatMessageSystem

# Paths
CURRENT_DIR = Path(__file__).parent.parent
EVAL_DATASET_PATH = CURRENT_DIR / "local_datasets/dpo_eval.jsonl"

# Refusal patterns (you implemented these in ex3!)
REFUSAL_PATTERNS = [
    r"I can only (help|answer|assist) with",
    r"questions about Alex",
    r"Alex('s| Loftus)",
    r"background, experience, skills",
    r"I'm sorry,? but I can('t| not)",
    r"outside (of )?my (scope|knowledge)",
    r"not related to",
    r"politely decline",
]


# =============================================================================
# Reuse components from previous exercises
# =============================================================================

@scorer(metrics=[accuracy()])
def refusal_detector():
    """Detect if model output contains refusal patterns (from ex3)."""
    async def score(state: TaskState, target: Target) -> Score:
        output_text = state.output.completion
        is_refusal = any(
            re.search(pattern, output_text, re.IGNORECASE)
            for pattern in REFUSAL_PATTERNS
        )
        return Score(
            value="C" if is_refusal else "I",
            answer=output_text[:200],
            explanation="Refusal detected" if is_refusal else "No refusal",
        )
    return score


@solver
def with_resume_context():
    """Inject system prompt and resume context (from ex5)."""
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        if RESUME:
            state.messages.insert(0, ChatMessageSystem(content=RESUME))
        if SYSTEM_PROMPT:
            state.messages.insert(0, ChatMessageSystem(content=SYSTEM_PROMPT))
        return await generate(state)
    return solve


def record_to_sample(record: dict) -> Sample:
    """Convert JSONL record to Sample (from ex2)."""
    return Sample(
        input=record["prompt"],
        target=record["chosen"],
        metadata={"rejected": record["rejected"]},
    )


@task
def dpo_eval():
    """Full DPO evaluation task combining all previous exercises."""
    dataset = json_dataset(str(EVAL_DATASET_PATH), sample_fields=record_to_sample)
    return Task(
        dataset=dataset,
        solver=with_resume_context(),
        scorer=refusal_detector(),
    )


# =============================================================================
# Main comparison function
# =============================================================================

def run_comparison(limit: int = 20):
    """
    Run evaluation on both base and DPO models, compare results.

    This is the payoff! You'll see:
    - Base model refusal rate
    - DPO model refusal rate
    - Improvement from DPO training
    """
    print("=" * 60)
    print("DPO Model Comparison")
    print("=" * 60)
    print(f"Base model: {MODEL_NAME}")
    print(f"DPO checkpoint: {CHECKPOINT_PATH}")
    print(f"Samples: {limit}")
    print()

    # TODO: Run eval on base model (no checkpoint)
    # HINT: base_results = inspect_eval(
    #     dpo_eval(),
    #     model=f"peft/{MODEL_NAME}",
    #     limit=limit,
    # )
    base_results = None

    # TODO: Run eval on DPO model (with checkpoint)
    # HINT: dpo_results = inspect_eval(
    #     dpo_eval(),
    #     model=f"peft/{MODEL_NAME}",
    #     model_args={"checkpoint_path": CHECKPOINT_PATH},
    #     limit=limit,
    # )
    dpo_results = None

    # TODO: Extract and compare accuracy (refusal rate)
    # HINT:
    # base_accuracy = base_results[0].results.scores[0].metrics["accuracy"].value
    # dpo_accuracy = dpo_results[0].results.scores[0].metrics["accuracy"].value
    #
    # print()
    # print("=" * 60)
    # print("Results")
    # print("=" * 60)
    # print(f"Base model refusal rate: {base_accuracy:.1%}")
    # print(f"DPO model refusal rate:  {dpo_accuracy:.1%}")
    # print(f"Improvement:             {dpo_accuracy - base_accuracy:+.1%}")

    print("\n[TODO: Implement the comparison logic above]")


if __name__ == "__main__":
    print("Exercise 6: DPO Model Comparison")
    print()
    print("This exercise brings everything together!")
    print()
    print("The PeftModelAPI is already implemented in model_api.py.")
    print("Your job: use inspect_eval() to run both models and compare.")
    print()
    print("After completing the TODOs in run_comparison():")
    print("  uv run python ex6_dpo_comparison.py")
    print()

    # Uncomment after implementing:
    # run_comparison(limit=20)
