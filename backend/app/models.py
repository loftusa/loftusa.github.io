"""SQLAlchemy 2.0 ORM models — the SQLite schema that replaces the JSONL "database".

Tables:
  coauthorship_events / affiliation_events  append-only crowd-edit logs, keyed on the
                                            microsecond-ISO `ts` (the LWW + delete key).
  chat_conversations / chat_messages        durable conversation store (cost cap survives restart).
  chat_logs                                 analytics log (mirror of the old chat_logs.jsonl row).
  users                                     account foundation for P3 (NextAuth upserts here).
  rate_limit_hits                           sliding-window limiter rows (restart-proof).
  migration_state                           bookkeeping for the JSONL→SQLite importer / cutover.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import JSON, DateTime, Float, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


def _utcnow() -> datetime:
    # naive UTC — identical string form to the old datetime.utcnow(), so imported JSONL `ts`
    # values and freshly-minted ones sort together (no "+00:00" suffix divergence).
    return datetime.now(timezone.utc).replace(tzinfo=None)


class _EventMixin:
    """Shared shape for the two crowd-edit event logs. `ts` is unique so DELETE ?ts= targets one
    logical event and the JSONL importer can INSERT-OR-IGNORE idempotently."""

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ts: Mapped[str] = mapped_column(String(32), nullable=False, unique=True, index=True)
    type: Mapped[str] = mapped_column(String(48), nullable=False)
    payload: Mapped[dict] = mapped_column(JSON, nullable=False)
    editor: Mapped[Optional[str]] = mapped_column(String(160))
    note: Mapped[Optional[str]] = mapped_column(String(600))
    ip: Mapped[Optional[str]] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=_utcnow, nullable=False
    )

    def to_event(self) -> dict:
        """Canonical event dict — exactly what fold_events / the raw JSONL export consume."""
        return {
            "type": self.type,
            "payload": self.payload,
            "editor": self.editor,
            "note": self.note,
            "ts": self.ts,
            "ip": self.ip,
        }


class CoauthorshipEvent(_EventMixin, Base):
    __tablename__ = "coauthorship_events"


class AffiliationEvent(_EventMixin, Base):
    __tablename__ = "affiliation_events"


class ChatConversation(Base):
    __tablename__ = "chat_conversations"

    user_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    cost_usd: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    message_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=_utcnow, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=_utcnow, onupdate=_utcnow, nullable=False
    )


class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(String(64), index=True, nullable=False)
    role: Mapped[str] = mapped_column(String(16), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=_utcnow, nullable=False
    )


class ChatLog(Base):
    __tablename__ = "chat_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[Optional[str]] = mapped_column(String(64), index=True)
    user_message: Mapped[str] = mapped_column(Text, nullable=False)
    bot_response: Mapped[str] = mapped_column(Text, nullable=False)
    token_latency_ms: Mapped[float] = mapped_column(Float, default=-1.0)
    ts: Mapped[str] = mapped_column(String(32), nullable=False)
    message_count: Mapped[int] = mapped_column(Integer, default=0)
    token_count: Mapped[int] = mapped_column(Integer, default=0)
    conversation_cost_usd: Mapped[float] = mapped_column(Float, default=0.0)
    rag_chunks_count: Mapped[int] = mapped_column(Integer, default=0)
    rag_sources: Mapped[list] = mapped_column(JSON, default=list)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=_utcnow, nullable=False
    )


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    email: Mapped[str] = mapped_column(
        String(255), unique=True, index=True, nullable=False
    )
    name: Mapped[Optional[str]] = mapped_column(String(255))
    provider: Mapped[Optional[str]] = mapped_column(String(40))  # "github" | "google"
    provider_sub: Mapped[Optional[str]] = mapped_column(
        String(255)
    )  # provider's stable subject id
    role: Mapped[str] = mapped_column(String(20), default="user", nullable=False)
    session_version: Mapped[int] = mapped_column(
        Integer, default=1, nullable=False
    )  # hard-revoke
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=_utcnow, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=_utcnow, onupdate=_utcnow, nullable=False
    )


class RateLimitHit(Base):
    __tablename__ = "rate_limit_hits"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    client_ip: Mapped[str] = mapped_column(String(64), index=True, nullable=False)
    ts: Mapped[float] = mapped_column(
        Float, index=True, nullable=False
    )  # epoch seconds
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=_utcnow, nullable=False
    )


class SpendEvent(Base):
    """One row per metered LLM turn (USD). Summed over a trailing 24h window to enforce
    DAILY_COST_CEILING_USD (previously config-only, never checked); pruned by housekeeping
    after SPEND_RETENTION_SECONDS."""

    __tablename__ = "spend_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ts: Mapped[float] = mapped_column(
        Float, index=True, nullable=False
    )  # epoch seconds
    usd: Mapped[float] = mapped_column(Float, nullable=False)


class MigrationState(Base):
    __tablename__ = "migration_state"

    key: Mapped[str] = mapped_column(String(80), primary_key=True)
    value: Mapped[str] = mapped_column(Text, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=_utcnow, onupdate=_utcnow, nullable=False
    )


class KlistSubmission(Base):
    """Filled /klist checklists from potential partners. Private: write-only for
    visitors, read only via the bearer-gated GET (see routers/klist.py); rows live
    on the Fly volume, never in the public repo."""

    __tablename__ = "klist_submissions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ts: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    name: Mapped[Optional[str]] = mapped_column(String(200))
    payload: Mapped[dict] = mapped_column(JSON, nullable=False)
    ip: Mapped[Optional[str]] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=_utcnow, nullable=False
    )


class KlistSchemaItem(Base):
    """Visitor-added sections/tiles for the /klist checklist (the list evolves).

    Each row is one added item; a new section exists once its first item does.
    Anyone with the form link can add (rate-limited, deduped); only the admin
    PIN can delete. Base sections/items live in the static pages, not here."""

    __tablename__ = "klist_schema_items"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    section: Mapped[str] = mapped_column(String(80), nullable=False)
    item: Mapped[str] = mapped_column(String(120), nullable=False)
    ip: Mapped[Optional[str]] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=_utcnow, nullable=False
    )


class JobsInterest(Base):
    """Pro/dossier interest submissions from the /jobs board. Private: write-only
    for visitors, read only via the token-gated GET (see routers/jobs.py); rows
    live on the Fly volume, never in the public repo."""

    __tablename__ = "jobs_interest"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ts: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    email: Mapped[str] = mapped_column(String(320), nullable=False)
    name: Mapped[Optional[str]] = mapped_column(String(200))
    tier: Mapped[str] = mapped_column(String(16), nullable=False)  # pro | dossier
    target_roles: Mapped[Optional[str]] = mapped_column(Text)
    notes: Mapped[Optional[str]] = mapped_column(Text)
    ip: Mapped[Optional[str]] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=_utcnow, nullable=False
    )


class HouseReachedOut(Base):
    """Listings on the /houses page that Alex has reached out about.

    Keyed by the Craigslist listing URL (stable across the daily data.js rebuild,
    unlike the L-ids). Unauthenticated personal store — see routers/houses.py.
    """

    __tablename__ = "house_reached_out"

    url: Mapped[str] = mapped_column(Text, primary_key=True)
    title: Mapped[Optional[str]] = mapped_column(Text)
    message: Mapped[Optional[str]] = mapped_column(Text)
    channel: Mapped[Optional[str]] = mapped_column(String(16))  # email | text | listing
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=_utcnow, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=_utcnow, onupdate=_utcnow, nullable=False
    )
