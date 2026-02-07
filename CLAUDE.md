# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Jekyll-based academic website (loftusa.github.io) with integrated Python ML experiments, a FastAPI chat backend, and a Next.js stocks page. The repository serves multiple purposes:
1. Static academic website (Jekyll/GitHub Pages)
2. ML research experiments (PyTorch/Transformers fine-tuning)
3. Production chat API (FastAPI + Cerebras LLM)
4. Experimental stocks dashboard (Next.js)

## Development Commands

### Jekyll Site (Academic Website)
```bash
# Install dependencies
bundle install

# Run local development server
bundle exec jekyll liveserve
# Visit http://localhost:4000

# Development with draft posts
bundle exec jekyll serve --config _config.yml,_config.dev.yml
```

### Python Experiments
```bash
# Run any Python script (ALWAYS use this syntax)
uv run <script>.py

# Examples:
uv run experiments/chat_api.py
uv run experiments/run_eval.py
uv run experiments/finetune/dataset.py

# Do NOT use:
# python <script>.py
# uv run python <script>.py
```

### Chat API (FastAPI Backend)
```bash
# Run locally
cd experiments/
uv run uvicorn chat_api:app --reload --port 8000

# Download chat logs (requires LOG_ACCESS_TOKEN in .env)
curl -H "Authorization: Bearer $LOG_ACCESS_TOKEN" http://localhost:8000/logs/download

# Health check
curl http://localhost:8000/health
```

### Stocks Page (Next.js)
```bash
npm run dev
# Visit http://localhost:3000/stocks

# API endpoint
curl http://localhost:3000/api/stocks
```

## Architecture

### Chat System (`experiments/`)
**Flow**: User → Jekyll frontend (`_pages/chat.md`) → FastAPI backend (`chat_api.py`) → Cerebras API → Response stream

Key files:
- `chat_api.py`: FastAPI server with `/chat`, `/health`, `/logs/download` endpoints
- `system_prompt.txt`: System instructions for the chat model
- `resume.txt`: Resume context injected into every chat
- `logs/chat_logs.jsonl`: User interaction logs (user_message, bot_response, latencies)
- `run_eval.py`: Evaluation harness for chat accuracy (fact-checking against `eval_dataset.json`)
- `analyze_logs.py`: Log analysis and statistics
- `sync_logs.py`: Script to download logs from production

**Environment variables** (`.env` in `experiments/`):
- `CEREBRAS_API_KEY`: API key for Cerebras LLM
- `LOG_PATH`: Path to chat logs JSONL file
- `LOG_ACCESS_TOKEN`: Bearer token for accessing logs endpoint
- `EVAL_DATASET_PATH`: Path to evaluation dataset JSON
- `EVAL_OUTPUT_PATH`: Path to evaluation output JSON

### Fine-tuning System (`experiments/finetune/`)

Two training approaches exist: SFT (inverse-length weighted) and DPO (direct preference optimization).

#### SFT Training (Length-Weighted)
**Goal**: Fine-tune with inverse-length weighted loss to produce shorter responses

Architecture (see `finetune_plan.md` for detailed design):
- `dataset.py`: `ChatDataset` class that tokenizes chat logs, masks prompt tokens in labels with -100, returns `{input_ids, labels, response_length}`. Uses Qwen2-0.5B-Instruct as base model.
- `collator.py`: `DataCollatorWithLengths` for left-padding batches and stacking response lengths
- `trainer.py`: `LengthWeightedTrainer` (subclass of HuggingFace Trainer) - overrides `compute_loss` to weight loss inversely by response length
- `train.py`: Main SFT training script using DoRA (LoRA with weight decomposition)
- `inference.py`: Compare base vs LoRA model responses, shows token count difference

**Key implementation detail**: Loss is computed per-token with `reduction="none"`, averaged per-example, then weighted by `1.0 / response_length`. Weights are normalized to preserve learning rate scale.

#### DPO Training (Refusal Training)
**Goal**: Train model to refuse off-topic/inappropriate requests instead of complying

**DPO Data Generation Pipeline**:
- `dpo_data_gen.py`: Simple approach - uses Claude to rewrite responses more concisely, creates preference pairs where shorter=chosen
- `dpo_data_gen_agents.py`: Agent-based approach using OpenAI Agents SDK with three agents:
  - `RequestClassifier`: Categorizes requests as legitimate/off_topic/jailbreak/inappropriate
  - `ComplianceChecker`: Checks if model incorrectly complied with bad requests
  - `RefusalAgent`: Generates brief, professional refusals
  - Only generates DPO pairs when model incorrectly complied (chosen=refusal, rejected=bad_response)
- `run_dpo_variance.sh`: Runs data generation 10 times to measure variance
- `analyze_dpo_variance.py`: Analyzes agreement across runs using sentence embeddings (all-MiniLM-L6-v2), generates visualizations in `variance_analysis/`
- `combine_dpo_runs.py`: Combines multiple runs into consensus dataset based on agreement threshold (default 50%), uses shortest chosen and modal rejected

**DPO Training & Inference**:
- `dpo_train.py`: DPO training using TRL's `DPOTrainer` with LoRA on Qwen2-0.5B-Instruct. Trains on `local_datasets/dpo_combined.jsonl`
- `dpo_chat.py`: Interactive chat to test DPO checkpoints. Commands: `quit`, `base` (toggle adapter), `ckpt#` (load checkpoint N)
- `view_dpo.py`: Pretty-print DPO dataset with rich formatting

**Data sources**:
1. Real chat logs from `chat_logs.jsonl`
2. Synthetic Q&A pairs generated from `resume.txt`

### Jailbreak Evaluation System (`experiments/evals/`)

Evaluates the production Cerebras `gpt-oss-120b` chatbot against jailbreak attacks using the UK AISI [Inspect AI](https://inspect.ai-safety-institute.org.uk/) framework. Uses Cerebras's OpenAI-compatible API (`openai/gpt-oss-120b` with `OPENAI_BASE_URL=https://api.cerebras.ai/v1`).

**Core pattern**: `Task = Dataset + Solver(s) + Scorer`

**Dataset**: `jailbreak_dataset.jsonl` — 96 samples (63 jailbreak/refuse + 33 legitimate/answer) built by `build_dataset.py` from three sources: DPO jailbreak data, eval_dataset.json, and handwritten prompts. Categories: legitimate, prompt_injection, off_topic, creative_bypass, emotional, roleplay, authority, hypothetical, topic_drift.

**6 progressive exercises** (docstring-only templates — user fills in code):
| Exercise | Concepts | What You Build |
|----------|----------|----------------|
| `ex1_hello_world.py` | Task, MemoryDataset, Sample | 8 hardcoded samples, basic eval |
| `ex2_jailbreak_dataset.py` | json_dataset(), record_to_sample() | Load 96-sample JSONL dataset |
| `ex3_refusal_scorer.py` | @scorer, regex patterns, two-case logic | Category-aware refusal/answer scorer |
| `ex4_llm_judge.py` | get_model(), LLM-as-judge | Claude judges model behavior |
| `ex5_system_context.py` | @solver, ChatMessageSystem | Inject system prompt + resume |
| `ex6_jailbreak_eval.py` | inspect_eval() API, results extraction | Per-category jailbreak defense table |

Key files:
- `model_api.py`: Maps `CEREBRAS_API_KEY` → `OPENAI_API_KEY`, sets `OPENAI_BASE_URL`. Exports `SYSTEM_PROMPT`, `RESUME`, `MODEL` constants.
- `build_dataset.py`: Dataset builder (run once: `uv run build_dataset.py`)
- `solutions.py`: Complete reference implementations for all 6 exercises
- `run_eval.py`: CLI runner (`uv run run_eval.py ex1`, `uv run run_eval.py ex6 --use-solutions`)

Running:
```bash
cd experiments/evals
inspect eval solutions.py@context_eval --model openai/gpt-oss-120b --limit 10
uv run solutions.py 20  # full programmatic eval with per-category table
inspect view  # browse logs in browser
```

**Initial results** (15-sample test): 46.7% overall defense rate — creative_bypass 40%, emotional 100%, off_topic 50%, prompt_injection 33%.

### Jekyll Site Structure
- `_pages/`: Main pages (about, cv, chat, publications, etc.)
- `_posts/`: Blog posts (research notes, Bayesian analysis, etc.)
- `_drafts/`: Unpublished drafts
- `_publications/`: Academic publications metadata
- `_talks/`: Conference talks metadata
- `_includes/`, `_layouts/`, `_sass/`: Jekyll theme customization
- `_config.yml`: Site configuration (title, author, social links, etc.)
- `markdown_generator/`: Python scripts to generate markdown from TSV/BibTeX

### Next.js App (`app/`, `lib/`, `components/`)
- `app/stocks/`: Stocks dashboard page
- `app/api/stocks/`: API endpoint returning weekly stock changes
- `lib/holdings.ts`: Hard-coded holdings list (symbol, quantity)
- `lib/schwabSync.ts`: Stub for Schwab API sync (no-op, protected by `SYNC_TOKEN`)
- `middleware.ts`: Auth protection for admin endpoints

## Key Conventions

### Python Code Organization
- **Experiments go in `experiments/` directory** - not in root
- **Test scripts go in `tests/scratchpad/`** - don't clutter root
- **Use `uv run <script>.py`** for all Python execution
- When working with images models during tests, save outputs to `tests/scratchpad/` and open them to verify correctness

### Package Management
- Python: `uv` (see `pyproject.toml`)
- Jekyll: `bundle` (see `Gemfile`)
- Next.js: `npm` (see `package.json`)

### Environment Files
- `experiments/.env`: Chat API keys, log paths, eval paths
- Root `.env` (if exists): Fallback for backward compatibility

### Deployment
- Jekyll site: Automatically deployed via GitHub Pages
- Chat API: Deployed on Fly.io (see `fly.toml`, `Dockerfile` in experiments/)
- Stocks page: Next.js app directory (experimental)

## Important Notes

### Data Privacy
- Chat logs contain user messages - handle with care
- `LOG_ACCESS_TOKEN` required to download logs from API
- Holdings list in `lib/holdings.ts` is hard-coded (no PII intended for storage)

### Model Context
- Chat API injects **both** `system_prompt.txt` and `resume.txt` as system messages before every conversation
- Conversation history is passed in every request (not stored server-side)
- Model: Cerebras `gpt-oss-120b`

### Fine-tuning
- Uses Qwen2-0.5B-Instruct as base model (upgraded from TinyLlama)
- LoRA/PEFT for memory efficiency (DoRA for SFT, standard LoRA for DPO)
- Labels are masked with `-100` for prompt tokens (standard causal LM practice)
- SFT: Response length weighting (`weight = 1.0 / response_length`, normalized)
- DPO: Direct preference optimization for refusal training - model learns to refuse off-topic/inappropriate requests

### Testing
- Write tests in `tests/` directory (use `pytest` and `hypothesis`)
- Run tests regularly to ensure code correctness
- Update tests when editing existing code
- Fail fast: lots of assertions before expensive operations

### When Finishing Work
- After writing/updating experiments: update README.md and CLAUDE.md with what the experiment does
- After getting results: update README.md and CLAUDE.md with the results
- Update README.md if code changes warrant it (avoid unnecessary info)
