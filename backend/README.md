# backend — FastAPI + SQLite (Fly)

The always-warm API behind alex-loftus.com: the résumé RAG chatbot and the co-authorship /
affiliation crowd-edit maps. Replaces the old single-file `experiments/chat_api.py` (in git
history) and its JSONL "database" with a small package on a real SQLite DB.

## Why SQLite (vs the old JSONL)

- **Durable** rate-limits and per-conversation cost caps — they no longer reset to zero on every
  Fly redeploy (they used to live in process memory).
- A site-wide **daily cost ceiling** (`DAILY_COST_CEILING_USD`, default $20): `spend_events` rows
  are written per chat turn and summed over a trailing 24h in `/chat` before the LLM call. The
  per-conversation cap alone doesn't bound aggregate spend (fresh conversation ids are free).
- A `users` table as the foundation for accounts (P3).
- The crowd-edit logs stay **event-sourced**: the pure `fold_events` / `fold_aff_events` functions
  in `experiments/coauthorship/` are unchanged; only the event *source* moved from JSONL lines to
  DB rows. `GET …/overlay` is still `fold_events(events)`, byte-for-byte.

## Abuse guards (open endpoints)

- **Real client IP**: `deps.client_ip` prefers the `Fly-Client-IP` header (set — and overwritten —
  by Fly's edge proxy, so not client-spoofable). Without it, `request.client.host` on Fly is the
  proxy address and every visitor shares one rate-limit bucket.
- `/coauthorship|/affiliations /corrections`: `CORRECTION_RATE` per IP + payload size caps.
- `/translate`: `MAX_TRANSLATE_CHARS` input cap + `TRANSLATE_RATE` per IP (rate-limit keys are
  scoped `translate:<ip>` so the budgets don't mix).
- `/chat`: per-conversation cost cap + the daily ceiling above.
- On Fly (`FLY_APP_NAME` set) the dev-fallback values of `API_JWT_SECRET` / `INTERNAL_API_KEY`
  are disabled — a missing secret fails closed instead of verifying against a public string.

## Layout

```
backend/
  app/
    main.py          FastAPI assembly (CORS, routers, lifespan warmup)
    config.py        env-driven settings (DB url, paths, caps, pricing)
    db.py            SQLAlchemy engine (SQLite WAL + busy_timeout) + session factory
    models.py        8 tables (events ×2, conversations, messages, logs, users, rate_limit, migration_state)
    deps.py          require_bearer, client_ip
    routers/         chat, coauthorship, affiliations, admin  (HTTP layer + validation)
    services/        events, rate_limit, conversations, llm, rag  (logic; llm/rag are lazy)
    scripts/import_jsonl.py   idempotent JSONL → SQLite backfill
  migrations/        Alembic (env.py reads url+metadata from the app)
  alembic.ini  Dockerfile  entrypoint.sh  pyproject.toml
```

`experiments/` stays the library: pure fold logic, the RAG module + prebaked Chroma collection,
`system_prompt.txt`, `resume.txt`, `seeds.json`. The backend imports from it.

## Local dev

```bash
uv venv && source .venv/bin/activate
uv pip install -r backend/pyproject.toml --group dev      # add chromadb+cerebras to actually chat
python -m pytest backend/tests/ -q                        # 57 tests, no chromadb/cerebras needed
DATABASE_URL=sqlite:///dev.db uvicorn backend.app.main:app --reload
```

Tests use in-memory SQLite and lazy LLM/RAG, so they run without the heavy deps.

## Deploy + cutover runbook (Fly)

The DB lives on the existing `logs` volume at `/app/experiments/logs/app.db`. **Snapshot first.**

```bash
fly volumes list                                  # find the logs volume id
fly volumes snapshots create <vol_id>             # safety net before touching data

# 1. deploy the SQLite build with dual-write ON (keeps writing JSONL as a cold backup)
fly secrets set DUAL_WRITE_JSONL=1                 # CEREBRAS_API_KEY / LOG_ACCESS_TOKEN already set
fly deploy                                         # entrypoint runs `alembic upgrade head`

# 2. backfill the existing JSONL into SQLite (idempotent — safe to re-run)
fly ssh console -C "python -m backend.app.scripts.import_jsonl"

# 3. verify: overlay JSON unchanged + row counts match the JSONL line counts
curl -s https://<app>.fly.dev/coauthorship/overlay | head
fly ssh console -C "wc -l /app/experiments/logs/coauthorship_corrections.jsonl"

# 4. once happy, stop dual-writing (SQLite is now the source of truth)
fly secrets unset DUAL_WRITE_JSONL && fly deploy
```

Rollback: redeploy the previous image; the JSONL files are intact on the volume (and were
dual-written through step 3). The event `ts` is the unique key, so re-imports never duplicate.

## Env / secrets

| var | purpose |
|-----|---------|
| `CEREBRAS_API_KEY` | LLM (chat) |
| `LOG_ACCESS_TOKEN` | bearer for `/logs/download` + raw `…/corrections` export/delete |
| `DATA_DIR` | volume dir (set in fly.toml to `/app/experiments/logs`) |
| `DATABASE_URL` | optional override; defaults to `sqlite:///$DATA_DIR/app.db` |
| `DUAL_WRITE_JSONL` | `1` only during the cutover window |
