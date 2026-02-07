# Jailbreak Evaluation Exercises (Inspect AI)

Evaluate the production Cerebras `gpt-oss-120b` chatbot against jailbreak attacks using the UK AISI Inspect framework.

## Setup

```bash
# From experiments/evals/
# Build the jailbreak dataset (96 samples: 63 jailbreak + 33 legitimate)
uv run build_dataset.py

# Verify experiments/.env has OPENAI_API_KEY and OPENAI_BASE_URL set
# (Cerebras is OpenAI-compatible — we use Inspect's built-in openai/ provider)
# OPENAI_BASE_URL='https://api.cerebras.ai/v1'
# OPENAI_API_KEY='<your CEREBRAS_API_KEY>'
```

## Core Mental Model

```
Task = Dataset + Solver(s) + Scorer

Flow:
1. Dataset provides Samples (input + target + metadata)
2. Solver(s) transform TaskState (inject system context, call model)
3. Scorer evaluates output → Score ("C" correct / "I" incorrect)
```

Key object: **TaskState** holds `messages`, `output`, `input_text`, `metadata`

## Exercise Progression

| Exercise | Concepts | What You Build |
|----------|----------|----------------|
| ex1_hello_world.py | Task, MemoryDataset, Sample, generate(), includes() | 8 hardcoded samples, basic eval skeleton |
| ex2_jailbreak_dataset.py | json_dataset(), record_to_sample(), metadata | Load 96-sample JSONL dataset |
| ex3_refusal_scorer.py | @scorer, Score, regex patterns, two-case logic | Category-aware refusal/answer scorer |
| ex4_llm_judge.py | get_model(), LLM-as-judge, CORRECT/INCORRECT | Claude judges model behavior |
| ex5_system_context.py | @solver, ChatMessageSystem, state manipulation | Inject system prompt + resume |
| **ex6_jailbreak_eval.py** | **inspect_eval() API, results extraction** | **Per-category jailbreak defense table** |

## How to Work

1. Read the exercise file's docstring — it describes what to build and what imports to use
2. Write the implementation from scratch
3. Run the command shown in the docstring
4. Check against `solutions.py` if needed
5. Use `inspect view` to browse logs in the browser

## CLI Commands

```bash
cd experiments/evals

# Run exercises directly
inspect eval ex1_hello_world.py --model openai/gpt-oss-120b
inspect eval ex2_jailbreak_dataset.py --model openai/gpt-oss-120b --limit 10
inspect eval ex3_refusal_scorer.py --model openai/gpt-oss-120b --limit 20
inspect eval ex4_llm_judge.py --model openai/gpt-oss-120b --limit 10
inspect eval ex5_system_context.py --model openai/gpt-oss-120b --limit 10

# Exercise 6: standalone script
uv run ex6_jailbreak_eval.py
uv run ex6_jailbreak_eval.py 20  # limit to 20 samples

# Run solutions directly
inspect eval solutions.py@jailbreak_hello_world --model openai/gpt-oss-120b
inspect eval solutions.py@context_eval --model openai/gpt-oss-120b --limit 10
uv run solutions.py        # full eval with per-category table
uv run solutions.py 20     # limit

# Runner script
uv run run_eval.py ex1
uv run run_eval.py ex5 --limit 10
uv run run_eval.py ex6 --use-solutions

# Browse eval logs
inspect view
```

## Dataset

`jailbreak_dataset.jsonl` — 96 samples built by `build_dataset.py`:

| Category | Count | Source |
|----------|-------|--------|
| legitimate | 33 | eval_dataset.json |
| prompt_injection | 17 | DPO data + handwritten |
| off_topic | 16 | DPO data |
| creative_bypass | 13 | DPO data |
| emotional | 5 | DPO data + handwritten |
| roleplay | 4 | handwritten |
| authority | 3 | handwritten |
| hypothetical | 3 | handwritten |
| topic_drift | 2 | handwritten |

## Files

| File | Description |
|------|-------------|
| `model_api.py` | Sets up Cerebras as OpenAI-compatible provider for Inspect |
| `build_dataset.py` | Builds `jailbreak_dataset.jsonl` from 3 sources |
| `ex1-ex6` | Exercise description files (you write the code) |
| `solutions.py` | Complete reference implementations |
| `run_eval.py` | CLI runner for exercises |
| `jailbreak_dataset.jsonl` | 96-sample evaluation dataset |

## Verification Checklist

- [ ] Ex1: `inspect eval` runs, logs appear in `inspect view`
- [ ] Ex2: 96 samples loaded, metadata visible in logs
- [ ] Ex3: Accuracy metric appears, jailbreaks scored correctly
- [ ] Ex4: LLM judge produces CORRECT/INCORRECT scores
- [ ] Ex5: Model answers Alex questions, refuses jailbreaks
- [ ] **Ex6: Per-category defense rate table printed**
