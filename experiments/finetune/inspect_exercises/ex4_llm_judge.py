"""
Exercise 4: Model-Graded Scorer (LLM-as-Judge)

CONCEPT: Using a grader model to evaluate responses

YOUR GOAL:
1. Write a custom scorer that uses an LLM to judge model vs reference
2. Use get_model() to call a grader model
3. Parse the judgment (A/B/TIE) and return appropriate Score

RUN WHEN COMPLETE:
    inspect eval ex4_llm_judge.py --model peft/Qwen/Qwen2-0.5B-Instruct --limit 5

    Note: The grader model (Claude) is separate from the model being evaluated.
    Your local Qwen model generates responses, Claude judges them.

SUCCESS CRITERIA:
- LLM grader is called and produces judgments
- Scores are computed based on who wins

WHAT YOU'RE LEARNING: Model grading, async model calls, custom prompts
"""

from model_api import register_peft_model
register_peft_model()

from pathlib import Path

from inspect_ai import Task, task
from inspect_ai.dataset import Sample, json_dataset
from inspect_ai.solver import generate, TaskState
from inspect_ai.scorer import scorer, Score, Target, accuracy
from inspect_ai.model import get_model

# Path to evaluation dataset
EVAL_DATASET_PATH = Path(__file__).parent.parent / "local_datasets/dpo_eval.jsonl"

# Judge prompt template (from dpo_eval.py)
# This is provided for you - the judge compares Response A (model) vs Response B (reference)
JUDGE_TEMPLATE = """You are judging responses from a personal resume chatbot for Alexander Loftus.
The chatbot should:
1. Answer questions about Alex's background, experience, skills, research, and projects
2. Politely refuse off-topic or inappropriate requests
3. Keep responses concise and on-topic

User prompt: "{prompt}"

Response A (Model Output):
{model_response}

Response B (Reference):
{reference_response}

Which response is more appropriate for a personal resume chatbot?
Answer with ONLY one of: "A", "B", or "TIE"
"""


def record_to_sample(record: dict) -> Sample:
    """Convert a JSONL record to an Inspect Sample."""
    return Sample(
        input=record["prompt"],
        target=record["chosen"],
        metadata={"rejected": record["rejected"]},
    )


# TODO: Implement the llm_judge scorer
#
# HINTS:
# 1. Use @scorer(metrics=[accuracy()]) decorator
# 2. Accept grader_model parameter (default: "anthropic/claude-sonnet-4-20250514")
# 3. In the async score function:
#    a. Get grader with: grader = get_model(grader_model)
#    b. Format JUDGE_TEMPLATE with prompt, model_response, reference_response
#    c. Call: result = await grader.generate(judge_prompt)
#    d. Parse result.completion for A/B/TIE
#    e. A wins -> model wins -> value="C"
#    f. B wins -> reference wins -> value="I"
#    g. TIE -> value="C" (acceptable)

@scorer(metrics=[accuracy()])
def llm_judge(grader_model: str = "anthropic/claude-sonnet-4-20250514"):
    """Use an LLM to judge model output vs reference."""

    async def score(state: TaskState, target: Target) -> Score:
        # TODO: Get the grader model
        grader = None  # HINT: get_model(grader_model)

        # TODO: Build the judge prompt
        # HINT: state.input_text, state.output.completion, target.text
        judge_prompt = JUDGE_TEMPLATE.format(
            prompt=None,  # TODO
            model_response=None,  # TODO
            reference_response=None,  # TODO
        )

        # TODO: Call the grader
        result = None  # HINT: await grader.generate(judge_prompt)
        judgment = None  # HINT: result.completion.strip().upper()

        # TODO: Parse judgment and determine winner
        # if "A" in judgment and "B" not in judgment -> model wins
        # elif "B" in judgment and "A" not in judgment -> reference wins
        # else -> tie
        winner = None
        value = None

        return Score(
            value=value,
            answer=None,  # TODO: state.output.completion[:200]
            explanation=None,  # TODO: f"Judge chose: {winner}"
        )

    return score


@task
def judge_eval():
    """Evaluate using LLM-as-judge against reference responses."""
    dataset = json_dataset(
        str(EVAL_DATASET_PATH),
        sample_fields=record_to_sample,
    )

    return Task(
        dataset=dataset,
        solver=generate(),
        scorer=llm_judge(),
    )


if __name__ == "__main__":
    print("Exercise 4: Model-Graded Scorer (LLM-as-Judge)")
    print("\nKey concepts:")
    print("  - get_model(name) returns a model you can call")
    print("  - await model.generate(prompt) returns ModelOutput")
    print("  - result.completion contains the text response")
    print("\nAfter completing the TODOs, run:")
    print("  inspect eval ex4_llm_judge.py --model peft/Qwen/Qwen2-0.5B-Instruct --limit 5")
