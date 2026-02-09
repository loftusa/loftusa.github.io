"""
Inspect AI Evaluation Suite for DPO Resume Chatbot

Learning exercises to understand Inspect framework by porting dpo_eval.py metrics.

Run with:
    uv run python run_eval.py ex1 --limit 5        # Base model
    uv run python run_eval.py ex1 --dpo --limit 5  # DPO model

Exercises:
1. Hello World - basic Task/Solver/Scorer flow
2. Custom Dataset Loading - load full dpo_eval.jsonl
3. Custom Scorer - refusal detection with regex
4. Model-Graded Scorer - LLM-as-judge comparison
5. Custom Solver - system prompt injection
6. DPO Comparison - compare base vs DPO model
"""

# Register the PEFT model API
from model_api import register_peft_model
register_peft_model()

import re
from pathlib import Path

from inspect_ai import Task, task, eval
from inspect_ai.dataset import MemoryDataset, Sample, json_dataset
from inspect_ai.solver import generate, solver, Generate, TaskState
from inspect_ai.scorer import includes, scorer, Score, Target, accuracy
from inspect_ai.model import ChatMessageSystem, get_model

# Paths (adjust for being in inspect_exercises/ subfolder)
CURRENT_DIR = Path(__file__).parent.parent  # Up to finetune/
EVAL_DATASET_PATH = CURRENT_DIR / "local_datasets/dpo_eval.jsonl"
SYSTEM_PROMPT_PATH = CURRENT_DIR.parent / "system_prompt.txt"
RESUME_PATH = CURRENT_DIR.parent / "resume.txt"

# Load context files
SYSTEM_PROMPT = SYSTEM_PROMPT_PATH.read_text() if SYSTEM_PROMPT_PATH.exists() else ""
RESUME = RESUME_PATH.read_text() if RESUME_PATH.exists() else ""

# Refusal patterns (from dpo_eval.py)
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
# Exercise 1: Hello World
# =============================================================================
# Goal: Understand basic Task/Solver/Scorer flow
# - Load 5 hardcoded samples
# - Use generate() solver (just call the model)
# - Use includes() scorer to check if "Alex" appears in output

@task
def hello_world():
    """Basic hello world task - check if model mentions 'Alex'."""
    # Hardcoded samples for testing
    samples = [
        Sample(input="Who is Alex Loftus?", target="Alex"),
        Sample(input="What is Alex's education?", target="Alex"),
        Sample(input="Where did Alex go to school?", target="Alex"),
        Sample(input="What are Alex's skills?", target="Alex"),
        Sample(input="Tell me about Alex's work experience.", target="Alex"),
    ]

    return Task(
        dataset=MemoryDataset(samples),
        solver=generate(),
        scorer=includes(),
    )


# =============================================================================
# Exercise 2: Custom Dataset Loading
# =============================================================================
# Goal: Load full dpo_eval.jsonl (221 samples)
# - Map prompt -> input, chosen -> target
# - Store rejected in metadata

def record_to_sample(record: dict) -> Sample:
    """Convert a JSONL record to an Inspect Sample."""
    return Sample(
        input=record["prompt"],
        target=record["chosen"],
        metadata={"rejected": record["rejected"]},
    )


@task
def full_dataset():
    """Load full DPO eval dataset from JSONL."""
    dataset = json_dataset(
        str(EVAL_DATASET_PATH),
        sample_fields=record_to_sample,
    )

    return Task(
        dataset=dataset,
        solver=generate(),
        scorer=includes(),
    )


# =============================================================================
# Exercise 3: Custom Scorer - Refusal Detection
# =============================================================================
# Goal: Port the refusal detection from dpo_eval.py
# - Check if output contains any refusal pattern
# - Return Score(value="C") if refusal detected, "I" otherwise

@scorer(metrics=[accuracy()])
def refusal_detector():
    """Detect if model output contains refusal patterns."""
    async def score(state: TaskState, target: Target) -> Score:
        output_text = state.output.completion

        # Check for any refusal pattern
        is_refusal = any(
            re.search(pattern, output_text, re.IGNORECASE)
            for pattern in REFUSAL_PATTERNS
        )

        return Score(
            value="C" if is_refusal else "I",
            answer=output_text[:200],  # Truncate for readability
            explanation="Refusal pattern detected" if is_refusal else "No refusal pattern",
        )

    return score


@task
def refusal_eval():
    """Evaluate refusal rate on DPO dataset."""
    dataset = json_dataset(
        str(EVAL_DATASET_PATH),
        sample_fields=record_to_sample,
    )

    return Task(
        dataset=dataset,
        solver=generate(),
        scorer=refusal_detector(),
    )


# =============================================================================
# Exercise 4: Model-Graded Scorer (LLM-as-Judge)
# =============================================================================
# Goal: Compare model output against target (chosen response)
# - Have a grader model decide which is more appropriate

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


@scorer(metrics=[accuracy()])
def llm_judge(grader_model: str = "anthropic/claude-sonnet-4-20250514"):
    """Use an LLM to judge model output vs reference."""
    async def score(state: TaskState, target: Target) -> Score:
        grader = get_model(grader_model)

        # Build judge prompt
        judge_prompt = JUDGE_TEMPLATE.format(
            prompt=state.input_text,
            model_response=state.output.completion,
            reference_response=target.text,
        )

        # Call grader
        result = await grader.generate(judge_prompt)
        judgment = result.completion.strip().upper()

        # Parse judgment
        if "A" in judgment and "B" not in judgment:
            winner = "model"
            value = "C"  # Model wins = Correct
        elif "B" in judgment and "A" not in judgment:
            winner = "reference"
            value = "I"  # Reference wins = Incorrect
        else:
            winner = "tie"
            value = "C"  # Tie counts as correct (model is acceptable)

        return Score(
            value=value,
            answer=state.output.completion[:200],
            explanation=f"Judge chose: {winner} (raw: {judgment})",
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


# =============================================================================
# Exercise 5: Custom Solver - System Prompt Injection
# =============================================================================
# Goal: Inject system prompt and resume context before generation
# - Prepend system messages
# - Then call generate()

@solver
def with_resume_context():
    """Solver that injects system prompt and resume before generation."""
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        # Prepend system messages (insert at beginning, in reverse order)
        if RESUME:
            state.messages.insert(0, ChatMessageSystem(content=RESUME))
        if SYSTEM_PROMPT:
            state.messages.insert(0, ChatMessageSystem(content=SYSTEM_PROMPT))

        # Call model
        return await generate(state)

    return solve


@task
def context_eval():
    """Evaluate with system prompt and resume context injected."""
    # Sample questions that require resume knowledge
    samples = [
        Sample(input="What is Alex's education?", target="Northeastern"),
        Sample(input="Who is Alex's PhD advisor?", target="David Bau"),
        Sample(input="What textbook did Alex write?", target="Network"),
        Sample(input="What Kaggle competition did Alex win?", target="Vesuvius"),
        Sample(input="What is Alex's email?", target="alexloftus2004"),
    ]

    return Task(
        dataset=MemoryDataset(samples),
        solver=with_resume_context(),
        scorer=includes(),
    )


@task
def full_context_eval():
    """Full DPO eval with context injection and refusal detection."""
    dataset = json_dataset(
        str(EVAL_DATASET_PATH),
        sample_fields=record_to_sample,
    )

    return Task(
        dataset=dataset,
        solver=with_resume_context(),
        scorer=refusal_detector(),
    )


# =============================================================================
# Combined evaluation with multiple scorers
# =============================================================================

@task
def comprehensive_eval():
    """Run comprehensive evaluation with context and refusal detection."""
    dataset = json_dataset(
        str(EVAL_DATASET_PATH),
        sample_fields=record_to_sample,
    )

    return Task(
        dataset=dataset,
        solver=with_resume_context(),
        scorer=refusal_detector(),
    )


# =============================================================================
# Exercise 6: Custom ModelAPI for PEFT/LoRA models
# =============================================================================
# Goal: Evaluate your actual DPO-trained model vs base model
# - Create custom ModelAPI for PEFT
# - Run comparison between base and DPO

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from inspect_ai.model import (
    ChatMessageUser,
    GenerateConfig,
    ModelOutput,
    modelapi,
    ModelAPI,
)

# Model config
MODEL_NAME = "Qwen/Qwen2-0.5B-Instruct"
CHECKPOINT_DIR = CURRENT_DIR.parent / "checkpoints/qwen-instruct-lora-dpo"
CHECKPOINT_NUM = 126


@modelapi(name="peft")
class PeftModelAPI(ModelAPI):
    """Custom ModelAPI for PEFT/LoRA models."""

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        checkpoint_path: str | None = None,
        device: str = "cuda",
        **kwargs
    ):
        super().__init__(model_name, **kwargs)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, device_map=device
        )

        if checkpoint_path:
            self.model = PeftModel.from_pretrained(self.model, checkpoint_path)
            self.model.eval()

        self._model_name = model_name
        self._checkpoint_path = checkpoint_path

    async def generate(
        self,
        input: list,
        tools: list = [],
        tool_choice=None,
        config: GenerateConfig = GenerateConfig(),
        **kwargs
    ) -> ModelOutput:
        """Generate a response from the model."""
        # Convert Inspect messages to chat format
        messages = []
        for msg in input:
            if isinstance(msg, ChatMessageSystem):
                messages.append({"role": "system", "content": msg.content})
            elif isinstance(msg, ChatMessageUser):
                messages.append({"role": "user", "content": msg.content})
            else:
                messages.append({"role": "assistant", "content": msg.content})

        # Apply chat template
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Tokenize and generate
        inputs = self.tokenizer(text, return_tensors='pt').to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=config.max_tokens or 256,
                do_sample=True,
                temperature=config.temperature or 0.7,
            )

        # Decode response (skip input tokens)
        response = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )

        return ModelOutput.from_content(model=self._model_name, content=response)


def run_dpo_comparison(limit: int = 20):
    """Run evaluation comparing base model vs DPO model."""
    from inspect_ai import eval as inspect_eval

    checkpoint_path = str(CHECKPOINT_DIR / f"checkpoint-{CHECKPOINT_NUM}")

    print("=" * 60)
    print("DPO Model Comparison")
    print("=" * 60)
    print(f"Base model: {MODEL_NAME}")
    print(f"DPO checkpoint: {checkpoint_path}")
    print(f"Samples: {limit}")
    print()

    # Eval on base model
    print("Evaluating base model...")
    base_results = inspect_eval(
        full_context_eval(),
        model=f"peft/{MODEL_NAME}",
        limit=limit,
    )

    # Eval on DPO model
    print("Evaluating DPO model...")
    dpo_results = inspect_eval(
        full_context_eval(),
        model=f"peft/{MODEL_NAME}",
        model_args={"checkpoint_path": checkpoint_path},
        limit=limit,
    )

    # Extract and compare
    base_acc = base_results[0].results.scores[0].metrics["accuracy"].value
    dpo_acc = dpo_results[0].results.scores[0].metrics["accuracy"].value

    print()
    print("=" * 60)
    print("Results")
    print("=" * 60)
    print(f"Base model refusal rate: {base_acc:.1%}")
    print(f"DPO model refusal rate:  {dpo_acc:.1%}")
    print(f"Improvement:             {dpo_acc - base_acc:+.1%}")


if __name__ == "__main__":
    print("Inspect AI Solutions - DPO Resume Chatbot Evaluation")
    print()
    print("For exercises 1-5 (API models):")
    print("  inspect eval solutions.py:hello_world --model openai/gpt-4o-mini")
    print()
    print("For exercise 6 (local DPO model comparison):")
    print("  uv run python solutions.py --compare")
    print()
    print("Available tasks:")
    print("  - hello_world: Basic test (5 samples)")
    print("  - full_dataset: Full DPO eval (221 samples)")
    print("  - refusal_eval: Refusal detection scorer")
    print("  - judge_eval: LLM-as-judge scorer")
    print("  - context_eval: With system prompt injection (5 samples)")
    print("  - full_context_eval: Full eval with context + refusal")
    print()

    import sys
    if "--compare" in sys.argv:
        run_dpo_comparison(limit=20)
