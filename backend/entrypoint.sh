#!/usr/bin/env bash
# Bring the SQLite schema on the mounted volume up to date, then serve.
set -euo pipefail
cd /app

alembic -c backend/alembic.ini upgrade head

exec uvicorn backend.app.main:app --host 0.0.0.0 --port 8080
