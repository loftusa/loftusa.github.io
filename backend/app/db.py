"""SQLAlchemy engine + session factory for SQLite-on-a-volume.

WAL + busy_timeout make the single-writer SQLite tolerant of the always-warm Fly machine's
concurrent requests. In tests DATABASE_URL is `sqlite://` (in-memory, shared via StaticPool).
"""
from __future__ import annotations

from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from . import config
from .models import Base

_IN_MEMORY = config.DATABASE_URL in ("sqlite://", "sqlite:///:memory:")

_engine_kwargs: dict = {"future": True, "connect_args": {"check_same_thread": False}}
if _IN_MEMORY:
    _engine_kwargs[
        "poolclass"
    ] = StaticPool  # one connection so all sessions see the same DB

engine = create_engine(config.DATABASE_URL, **_engine_kwargs)


@event.listens_for(engine, "connect")
def _sqlite_pragmas(dbapi_conn, _record):
    cur = dbapi_conn.cursor()
    cur.execute("PRAGMA journal_mode=WAL")
    cur.execute("PRAGMA busy_timeout=5000")
    cur.execute("PRAGMA foreign_keys=ON")
    cur.close()


SessionLocal = sessionmaker(
    bind=engine, autoflush=False, expire_on_commit=False, future=True
)


def init_db() -> None:
    """Create tables (used at startup and in tests). Alembic owns prod migrations; this is the
    idempotent backstop so a fresh volume / dev box just works."""
    if not _IN_MEMORY:
        config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    Base.metadata.create_all(engine)


def get_db():
    """FastAPI dependency: a request-scoped session."""
    s = SessionLocal()
    try:
        yield s
    finally:
        s.close()
