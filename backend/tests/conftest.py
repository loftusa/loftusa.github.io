"""Shared test fixtures for the backend package.

In-memory SQLite (StaticPool → one shared connection) + a temp data dir, wired up
BEFORE importing backend.app so the engine binds to the test DB. Each test gets a
freshly-created schema for isolation.
"""
import os
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]  # repo root: holds backend/ and experiments/
sys.path.insert(0, str(ROOT))

# bind env BEFORE importing the app (paths/engine bind at import)
os.environ.setdefault("DATABASE_URL", "sqlite://")  # in-memory, shared via StaticPool
os.environ.setdefault("DATA_DIR", tempfile.mkdtemp())
os.environ.setdefault("LOG_ACCESS_TOKEN", "test-token")
os.environ.setdefault("KLIST_ACCESS_TOKEN", "test-klist-token")
os.environ.setdefault("JOBS_ACCESS_TOKEN", "test-jobs-token")
os.environ.setdefault("CEREBRAS_API_KEY", "test-key")

from backend.app import models  # noqa: E402
from backend.app.db import SessionLocal, engine  # noqa: E402


@pytest.fixture(autouse=True)
def clean_db():
    """Fresh schema per test."""
    models.Base.metadata.drop_all(engine)
    models.Base.metadata.create_all(engine)
    yield


@pytest.fixture
def session():
    s = SessionLocal()
    try:
        yield s
    finally:
        s.close()
