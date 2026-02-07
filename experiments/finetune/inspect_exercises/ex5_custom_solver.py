"""
Exercise 5: Custom Solver - System Prompt Injection

Goal: Write a solver that injects system prompt and resume context before generation

You need to:
1. Write a @solver function that modifies state.messages
2. Insert ChatMessageSystem objects at the beginning
3. Call await generate(state) and return the result

Run: uv run python run_eval.py ex5 --limit 5

Imports you'll need:
    from inspect_ai import Task, task
    from inspect_ai.dataset import MemoryDataset, Sample
    from inspect_ai.solver import solver, Generate, TaskState
    from inspect_ai.scorer import includes
    from inspect_ai.model import ChatMessageSystem

Solver pattern:
    @solver
    def my_solver():
        async def solve(state: TaskState, generate: Generate) -> TaskState:
            state.messages.insert(0, ChatMessageSystem(content="..."))
            return await generate(state)
        return solve

Context files available from model_api:
    from model_api import SYSTEM_PROMPT, RESUME
"""

from model_api import register_peft_model, SYSTEM_PROMPT, RESUME
register_peft_model()

# Your code here
