"""
Exercise 5: Custom Solver - System Prompt Injection

CONCEPT: Modifying TaskState before generation

YOUR GOAL:
1. Create a solver that prepends system prompt and resume context
2. Insert ChatMessageSystem objects into state.messages
3. Call generate() to get the model response

RUN WHEN COMPLETE:
    inspect eval ex5_custom_solver.py --model peft/Qwen/Qwen2-0.5B-Instruct

SUCCESS CRITERIA:
- Model receives context and can answer questions about Alex
- context_eval shows 100% accuracy on resume questions

WHAT YOU'RE LEARNING: Solver pipeline, state modification, message manipulation

KEY INSIGHT: Solvers transform TaskState. You can:
- Add/modify messages
- Set metadata
- Chain multiple solvers together
"""

from model_api import register_peft_model, SYSTEM_PROMPT, RESUME
register_peft_model()

from pathlib import Path

from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.solver import solver, Generate, TaskState
from inspect_ai.scorer import includes
from inspect_ai.model import ChatMessageSystem

# SYSTEM_PROMPT and RESUME are imported from model_api


# TODO: Implement the with_resume_context solver
#
# HINTS:
# 1. Use @solver decorator
# 2. Define async function: async def solve(state: TaskState, generate: Generate) -> TaskState
# 3. Insert system messages at beginning of state.messages:
#    state.messages.insert(0, ChatMessageSystem(content=RESUME))
#    state.messages.insert(0, ChatMessageSystem(content=SYSTEM_PROMPT))
# 4. Call and return: await generate(state)

@solver
def with_resume_context():
    """Solver that injects system prompt and resume before generation."""

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        # TODO: Insert resume context as a system message
        # HINT: state.messages.insert(0, ChatMessageSystem(content=RESUME))

        # TODO: Insert system prompt as a system message (before resume)
        # HINT: state.messages.insert(0, ChatMessageSystem(content=SYSTEM_PROMPT))

        # TODO: Call generate and return the result
        # HINT: return await generate(state)
        pass

    return solve


@task
def context_eval():
    """Evaluate with system prompt and resume context injected.

    These questions require knowledge from the resume to answer correctly.
    The includes() scorer checks if expected keywords appear in output.
    """
    samples = [
        Sample(input="What is Alex's education?", target="Northeastern"),
        Sample(input="Who is Alex's PhD advisor?", target="David Bau"),
        Sample(input="What textbook did Alex write?", target="Network"),
        Sample(input="What Kaggle competition did Alex win?", target="Vesuvius"),
        Sample(input="What is Alex's email?", target="alexloftus2004"),
    ]

    return Task(
        dataset=MemoryDataset(samples),
        solver=with_resume_context(),  # <-- Your custom solver!
        scorer=includes(),
    )


if __name__ == "__main__":
    print("Exercise 5: Custom Solver - System Prompt Injection")
    print(f"\nContext files loaded:")
    print(f"  System prompt: {len(SYSTEM_PROMPT)} chars")
    print(f"  Resume: {len(RESUME)} chars")
    print("\nKey concept: Solvers transform TaskState before generation")
    print("  - Insert messages: state.messages.insert(0, ChatMessageSystem(...))")
    print("  - Call model: await generate(state)")
    print("\nAfter completing the TODOs, run:")
    print("  inspect eval ex5_custom_solver.py --model peft/Qwen/Qwen2-0.5B-Instruct")
