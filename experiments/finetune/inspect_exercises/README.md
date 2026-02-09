# Inspect AI Learning Exercises

Learn the UK AISI Inspect framework by rebuilding your DPO evaluation suite.

## The Goal

By the end of these exercises, you'll have built an evaluation system that:
1. Loads your `dpo_eval.jsonl` dataset
2. Evaluates your DPO-trained model (from `dpo_train.py`)
3. Compares refusal rates between base Qwen and your DPO checkpoint
4. Replaces manual testing in `dpo_chat.py` with automated eval

## Core Mental Model

```
Task = Dataset + Solver(s) + Scorer

Flow:
1. Dataset provides Samples (input + target)
2. Solver(s) transform TaskState (add messages, call model)
3. Scorer evaluates output → Score
```

Key insight: Everything flows through **TaskState**, which holds:
- `messages`: conversation history
- `output`: model's final output
- `input_text`: original prompt
- `target`: expected answer/grading criteria

## Exercise Progression

All exercises use your local Qwen model via `model_api.py`.

| Exercise | Concept | What You Build |
|----------|---------|----------------|
| ex1_hello_world.py | Task, Dataset, basic Scorer | Basic eval skeleton |
| ex2_custom_dataset.py | Loading JSONL, field mapping | Load dpo_eval.jsonl |
| ex3_refusal_scorer.py | Custom scorer with regex | Refusal rate metric |
| ex4_llm_judge.py | Model-graded scorer | LLM-as-judge comparison |
| ex5_custom_solver.py | Custom solver | System prompt injection |
| **ex6_dpo_comparison.py** | **Programmatic eval** | **Compare base vs DPO!** |

## How to Work Through

1. Open an exercise file
2. Read the docstring for context
3. Fill in the TODOs (hints are provided)
4. Run to verify (command in each file's docstring)
5. Use `inspect view` to explore logs

## CLI Commands

```bash
# From this directory
cd experiments/finetune/inspect_exercises

# Run any exercise with your base Qwen model
inspect eval ex1_hello_world.py --model peft/Qwen/Qwen2-0.5B-Instruct

# Run with your DPO-trained model
inspect eval ex1_hello_world.py --model peft/Qwen/Qwen2-0.5B-Instruct \
    --model-args checkpoint_path=../checkpoints/qwen-instruct-lora-dpo/checkpoint-126

# Limit samples for faster iteration
inspect eval ex3_refusal_scorer.py --model peft/Qwen/Qwen2-0.5B-Instruct --limit 10

# View logs in browser
inspect view

# Exercise 6: Full comparison (both models)
uv run python ex6_dpo_comparison.py
```

## Verification Checklist

- [ ] Exercise 1: `inspect eval` runs, logs appear
- [ ] Exercise 2: 221 samples loaded correctly
- [ ] Exercise 3: Refusal accuracy metric appears
- [ ] Exercise 4: LLM judge produces scores
- [ ] Exercise 5: Model answers "What is Alex's education?" correctly
- [ ] **Exercise 6: See refusal rate comparison (base vs DPO)**

## Files

- `model_api.py` - Custom PEFT ModelAPI (already implemented)
- `ex1-ex6` - Exercise files with TODOs
- `solutions.py` - Complete solutions to check against

## Interview Talking Points

After completing these exercises, you can discuss:

1. **Architecture**: "Inspect uses a Task abstraction that combines Dataset, Solver chain, and Scorer."

2. **Solvers**: "Solvers are composable transformations on TaskState. You chain them like `[system_message(), generate()]`."

3. **Scorers**: "Scorers evaluate model output against targets. You can use built-ins like `includes()` or write custom async functions."

4. **Custom Models**: "I integrated my local PEFT/LoRA model by implementing a custom ModelAPI class."

5. **Practical Use**: "I used Inspect to compare my DPO-finetuned model against the base model, measuring refusal rate improvement."
