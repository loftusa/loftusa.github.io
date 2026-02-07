# Inspect AI Exercises

Build an evaluation system for your DPO model using the UK AISI Inspect framework.

## Exercises

| # | Goal | Run |
|---|------|-----|
| 1 | Basic Task + Dataset + Scorer | `uv run python run_eval.py ex1 --limit 5` |
| 2 | Load JSONL dataset | `uv run python run_eval.py ex2 --limit 5` |
| 3 | Custom scorer (regex) | `uv run python run_eval.py ex3 --limit 10` |
| 4 | LLM-as-judge scorer | `uv run python run_eval.py ex4 --limit 5` |
| 5 | Custom solver (system prompt) | `uv run python run_eval.py ex5 --limit 5` |
| 6 | Compare base vs DPO | `uv run python ex6_dpo_comparison.py` |

## Reference

`solutions.py` has working implementations.

## Quick Reference

```python
# Task = Dataset + Solver + Scorer
Task(dataset=my_dataset, solver=generate(), scorer=includes())

# Sample
Sample(input="question", target="expected", metadata={...})

# Scorer
@scorer(metrics=[accuracy()])
def my_scorer():
    async def score(state: TaskState, target: Target) -> Score:
        return Score(value="C" or "I", answer=state.output.completion)
    return score

# Solver
@solver
def my_solver():
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        state.messages.insert(0, ChatMessageSystem(content="..."))
        return await generate(state)
    return solve
```
