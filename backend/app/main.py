"""FastAPI application assembly. Vercel calls these endpoints cross-origin; Fly runs this image.

The app is intentionally thin: routers hold the endpoints, services hold the logic, db.py owns the
SQLite engine. RAG/LLM warm up lazily at startup (best-effort, never blocks boot).
"""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from . import config
from .db import SessionLocal, init_db
from .routers import (
    admin,
    affiliations,
    auth,
    chat,
    coauthorship,
    houses,
    klist,
    search,
    translate,
    ws,
)
from .services import housekeeping, rag


async def _housekeeping_loop() -> None:
    """Periodic durable housekeeping: prune old rate-limit rows, evict stale conversations."""
    while True:
        await asyncio.sleep(config.HOUSEKEEPING_INTERVAL_SECONDS)
        try:
            session = SessionLocal()
            try:
                print(f"housekeeping: {housekeeping.run_once(session)}")
            finally:
                session.close()
        except Exception as e:
            print(f"housekeeping error: {type(e).__name__}: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()  # idempotent backstop; Alembic owns real migrations in prod
    try:
        rag.warmup()  # pre-load embedding model to avoid cold-start latency
        print(f"RAG ready: {rag.count()} chunks indexed")
    except Exception as e:  # never let a RAG hiccup stop the server booting
        print(f"RAG warmup skipped: {type(e).__name__}: {e}")
    task = asyncio.create_task(_housekeeping_loop())
    try:
        yield
    finally:
        task.cancel()


def create_app() -> FastAPI:
    app = FastAPI(title="alex-loftus.com API", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.CORS_ORIGINS,
        allow_origin_regex=config.CORS_ORIGIN_REGEX,
        # Bearer-only API: no frontend fetch sends cookies, so never reflect a
        # credentialed origin (Authorization is still allowed via allow_headers).
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(chat.router)
    app.include_router(coauthorship.router)
    app.include_router(affiliations.router)
    app.include_router(admin.router)
    app.include_router(auth.router)
    app.include_router(translate.router)
    app.include_router(search.router)
    app.include_router(houses.router)
    app.include_router(klist.router)
    app.include_router(ws.router)
    return app


app = create_app()
