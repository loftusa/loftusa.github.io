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

# --- auth (P3) ---------------------------------------------------------------------------------
# HS256 bearer JWT minted by the Next.js BFF from the NextAuth session; verified here.
# The dev-fallback secrets are strings that live in this public repo, so on Fly they are
# disabled: a missing secret must fail closed (500/401), never verify against a known value.
_ON_FLY = bool(os.getenv("FLY_APP_NAME"))
API_JWT_SECRET = os.getenv("API_JWT_SECRET") or (
    None if _ON_FLY else "dev-insecure-jwt-secret"
)
JWT_ALGORITHM = "HS256"
JWT_TTL_SECONDS = int(os.getenv("JWT_TTL_SECONDS", "900"))  # 15 min
# S2S key gating the internal user-upsert endpoint (NextAuth signIn -> FastAPI).
INTERNAL_API_KEY = os.getenv("INTERNAL_API_KEY") or (
    None if _ON_FLY else "dev-insecure-internal-key"
)

# --- translate (P6) ----------------------------------------------------------------------------
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
TRANSLATE_MODEL = os.getenv("TRANSLATE_MODEL", "claude-haiku-4-5")
MAX_TRANSLATE_CHARS = 4000  # input cap — bounds per-call Anthropic spend
TRANSLATE_RATE = (20, 600)  # at most 20 translations / 600s / client IP

# --- housekeeping (P4) -------------------------------------------------------------------------
HOUSEKEEPING_INTERVAL_SECONDS = int(os.getenv("HOUSEKEEPING_INTERVAL_SECONDS", "3600"))
RATE_LIMIT_RETENTION_SECONDS = 7 * 24 * 3600  # prune rate-limit rows older than a week
DAILY_COST_CEILING_USD = float(os.getenv("DAILY_COST_CEILING_USD", "20"))
SPEND_RETENTION_SECONDS = 30 * 24 * 3600  # keep spend rows a month (cost audit trail)

# Extra CORS origins (comma-separated) + a regex for this project's Vercel PREVIEW deploys.
# Anyone can register aol-frontend-*.vercel.app as their own project name, so the regex is
# pinned to our team scope's deployment URLs (and credentials are off in main.py anyway).
_extra = os.getenv("EXTRA_CORS_ORIGINS", "")
CORS_ORIGIN_REGEX = (
    r"https://aol-frontend-[a-z0-9-]+-alexloftus2004-4021s-projects\.vercel\.app"
)

CORS_ORIGINS = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:4000",
    "http://127.0.0.1:4000",
    "http://127.0.0.1:3863",
    "https://alex-loftus.com",
    "https://www.alex-loftus.com",
    "https://aol-frontend-mu.vercel.app",  # the project's production alias
] + [o.strip() for o in _extra.split(",") if o.strip()]
