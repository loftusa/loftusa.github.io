"""
Exercise 2: Custom Dataset Loading (your data)

CONCEPT: Loading JSONL, mapping fields to Sample

YOUR GOAL:
1. Load dpo_eval.jsonl (221 samples) using json_dataset()
2. Write a record_to_sample() function that maps:
   - prompt -> input
   - chosen -> target
   - rejected -> metadata
3. Verify len(dataset) == 221

RUN WHEN COMPLETE:
    inspect eval ex2_custom_dataset.py --model peft/Qwen/Qwen2-0.5B-Instruct --limit 5

SUCCESS CRITERIA:
- Dataset loads 221 samples
- Each sample has input, target, and metadata["rejected"]

WHAT YOU'RE LEARNING: Data loading, field mapping, metadata storage
"""

from model_api import register_peft_model
register_peft_model()

from pathlib import Path

from inspect_ai import Task, task
from inspect_ai.dataset import Sample, json_dataset
from inspect_ai.solver import generate
from inspect_ai.scorer import includes

# Path to evaluation dataset (already set up for you)
EVAL_DATASET_PATH = Path(__file__).parent.parent / "local_datasets/dpo_eval.jsonl"


def record_to_sample(record: dict) -> Sample:
    """Convert a JSONL record to an Inspect Sample.

    TODO: Map the record fields to Sample fields:
    - record["prompt"] -> input (the question/request)
    - record["chosen"] -> target (the correct response)
    - record["rejected"] -> metadata (store for reference)

    HINT: Return Sample(
        input=...,
        target=...,
        metadata={"rejected": ...},
    )
    """
    # TODO: Implement this function
    pass


@task
def full_dataset():
    """Load full DPO eval dataset from JSONL."""

    # TODO: Load dataset using json_dataset()
    # HINT: json_dataset(str(path), sample_fields=record_to_sample)

    dataset = None  # TODO

    return Task(
        dataset=dataset,
        solver=generate(),
        scorer=includes(),
    )


if __name__ == "__main__":
    print("Exercise 2: Custom Dataset Loading")
    print(f"\nDataset path: {EVAL_DATASET_PATH}")

    # TODO: After implementing record_to_sample, uncomment to verify:
    # dataset = json_dataset(str(EVAL_DATASET_PATH), sample_fields=record_to_sample)
    # print(f"Loaded {len(dataset)} samples")
    # assert len(dataset) == 221, f"Expected 221, got {len(dataset)}"
    # print("SUCCESS!")

    print("\nAfter completing the TODOs, run:")
    print("  inspect eval ex2_custom_dataset.py --model peft/Qwen/Qwen2-0.5B-Instruct --limit 5")
