"""
Exercise 6: Full Jailbreak Evaluation — The Payoff

CONCEPTS: inspect_eval() programmatic API, results extraction, per-category breakdown

WHAT TO BUILD:
- Reuse (or import) components from ex2 (record_to_sample), ex3 (refusal_scorer),
  ex5 (with_resume_context) — or copy their implementations here
- A @task function called jailbreak_eval() combining all three into one Task
- An extract_category_results(results) function that:
    1. Gets eval log: log = results[0]
    2. Iterates log.samples
    3. Groups by sample.metadata["category"]
    4. Counts "C" values from sample.scores["refusal_scorer"].value
    5. Returns dict: category -> {"total": int, "defended": int, "rate": float}
- A print_results_table(category_results) function that prints:
    Category            | Samples | Defended | Rate
    --------------------+---------+----------+--------
    prompt_injection    |      17 |       15 |  88.2%
    ...
    legitimate          |      33 |       30 |  90.9%
    --------------------+---------+----------+--------
    OVERALL             |      96 |      78  |  81.3%
- A main() function that:
    1. Calls inspect_eval(jailbreak_eval(), model="openai/gpt-oss-120b", limit=limit)
    2. Calls extract_category_results + print_results_table

IMPORTS YOU'LL NEED:
    import model_api
    from model_api import SYSTEM_PROMPT, RESUME
    import re
    from pathlib import Path
    from collections import defaultdict
    from inspect_ai import eval as inspect_eval, Task, task
    from inspect_ai.dataset import Sample, json_dataset
    from inspect_ai.solver import solver, Generate, TaskState
    from inspect_ai.scorer import scorer, Score, Target, accuracy
    from inspect_ai.model import ChatMessageSystem

RUN:  uv run ex6_jailbreak_eval.py
      uv run ex6_jailbreak_eval.py 20    # limit to 20 samples
"""
from pathlib import Path

from model_api import SYSTEM_PROMPT, RESUME
from inspect_ai.solver import solver
from inspect_ai.dataset import json_dataset, Sample
from inspect_ai.model import ChatMessageSystem, GenerateConfig
from inspect_ai import task, Task

config = GenerateConfig(reasoning_effort=None)

# 1) get dataset loaded as an inspect dataset obj (record_to_sample)
DATASET_PATH = Path(__file__).parent / "jailbreak_dataset.jsonl"
assert DATASET_PATH.exists()

def record_to_sample(record: dict):
    prompt = record['prompt']
    target = record['target']
    metadata = {'category': record['category'], 'expected_behavior': record['expected_behavior']}
    return Sample(input=prompt, target=target, metadata=metadata)

jailbreak_dataset = json_dataset(
    str(DATASET_PATH),
    sample_fields=record_to_sample,
    name = "full jailbreaking dataset"
)

# 2) get a solver that inserts resume context into state
@solver
def with_resume_context():
    async def solve(state, generate_fn):
        state.messages.insert(0, ChatMessageSystem(content=SYSTEM_PROMPT))
        state.messages.insert(1, ChatMessageSystem(content=RESUME))
        return await generate_fn(state)
    return solve

@task
def jailbreak_eval():
    return Task(
        dataset=jailbreak_dataset,
        solver=with_resume_context(),
        scorer=None,
        config=config

    )

def print_results_table(results):
    raise NotImplementedError

import click
from inspect_ai import eval as inspect_eval

@click.command()
@click.option("--model", prompt="Model to use", help="Model to use.")
@click.option("--limit", default=None, help="Number of samples to evaluate.")
def main(model, limit):
    results = inspect_eval(jailbreak_eval(), model=model, limit=limit)
    print_results_table(results)

if __name__ == "__main__":
    main()