"""Central configuration for the FastAPI backend.

Everything is read from the environment at import time so tests can point the engine
and data dir at temp locations BEFORE importing the app. Defaults target the Fly
deployment (SQLite on the mounted volume at /app/experiments/logs).
"""
from __future__ import annotations

import os
from pathlib import Path

_HERE = Path(__file__).resolve()
REPO_ROOT = _HERE.parents[2]  # holds backend/ and experiments/
EXPERIMENTS = REPO_ROOT / "experiments"

# --- persistence -------------------------------------------------------------------------------
DATA_DIR = Path(
    os.getenv("DATA_DIR", "/app/experiments/logs")
)  # the Fly volume mount in prod
DATABASE_URL = os.getenv("DATABASE_URL") or f"sqlite:///{DATA_DIR / 'app.db'}"

# legacy JSONL files: the migration source, and (optionally) a dual-write cold backup
LOG_PATH = Path(os.getenv("LOG_PATH", str(DATA_DIR / "chat_logs.jsonl")))
CORRECTIONS_PATH = DATA_DIR / "coauthorship_corrections.jsonl"
AFF_EVENTS_PATH = DATA_DIR / "affiliation_events.jsonl"
DUAL_WRITE_JSONL = os.getenv("DUAL_WRITE_JSONL", "0") == "1"

LOG_ACCESS_TOKEN = os.getenv("LOG_ACCESS_TOKEN")

# --- crowd-edit guards (vandalism brakes; preserved from chat_api.py) --------------------------
MAX_CORRECTION_BYTES = 4096  # one event's JSON size cap
MAX_JOIN_EVENT_BYTES = 8192  # aff_join carries an entries array
MAX_EVENTS_PER_FAMILY = 50_000  # row-count cap (replaces the old 8MB JSONL file cap)
CORRECTION_RATE = (30, 600)  # at most 30 POSTs / 600s / client IP

# --- chat / LLM --------------------------------------------------------------------------------
CEREBRAS_API_KEY = os.getenv("CEREBRAS_API_KEY")
MODEL = os.getenv("CHAT_MODEL", "zai-glm-4.7")
INPUT_COST_PER_TOKEN = 2.25 / 1_000_000
OUTPUT_COST_PER_TOKEN = 2.75 / 1_000_000
MAX_COST_PER_CONVERSATION = 2.00  # USD
MAX_CONVERSATIONS = 1000  # LRU cap on persisted conversations
MAX_TURNS = 50  # user+assistant pairs kept per conversation

SYSTEM_PROMPT_PATH = EXPERIMENTS / "system_prompt.txt"
RESUME_PATH = EXPERIMENTS / "resume.txt"
AFF_SEEDS_PATH = EXPERIMENTS / "coauthorship" / "seeds.json"

CORS_ORIGINS = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:4000",
    "http://127.0.0.1:4000",
    "http://127.0.0.1:3863",
    "https://alex-loftus.com",
    "https://www.alex-loftus.com",
]
