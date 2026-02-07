"""
Exercise 5: System Context Solver

CONCEPTS: @solver, ChatMessageSystem, solver chaining, state manipulation

WHAT TO BUILD:
- A @solver function called with_resume_context() that:
    1. Inserts RESUME as ChatMessageSystem at position 0 in state.messages
    2. Inserts SYSTEM_PROMPT as ChatMessageSystem at position 0 (becomes first)
    3. Calls: return await generate(state)
    After inserts, message order is: [SYSTEM_PROMPT, RESUME, user_message]
- A @task function using jailbreak_dataset.jsonl with your solver + refusal_scorer from ex3

This mirrors what chat_api.py does: prepend system context before every request.
Without this solver, the model has no idea who Alex is and can't answer questions.

CONSTANTS (import from model_api):
    from model_api import SYSTEM_PROMPT, RESUME

KEY APIS:
    state.messages.insert(0, ChatMessageSystem(content=...))
    return await generate(state)

IMPORTS YOU'LL NEED:
    import model_api
    from model_api import SYSTEM_PROMPT, RESUME
    import re
    from pathlib import Path
    from inspect_ai import Task, task
    from inspect_ai.dataset import Sample, json_dataset
    from inspect_ai.solver import solver, Generate, TaskState
    from inspect_ai.scorer import scorer, Score, Target, accuracy
    from inspect_ai.model import ChatMessageSystem

RUN:  inspect eval ex5_system_context.py --model openai/gpt-oss-120b --limit 10
"""
