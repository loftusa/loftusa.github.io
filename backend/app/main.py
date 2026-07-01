"""FastAPI application assembly. Vercel calls these endpoints cross-origin; Fly runs this image.

The app is intentionally thin: routers hold the endpoints, services hold the logic, db.py owns the
SQLite engine. RAG/LLM warm up lazily at startup (best-effort, never blocks boot).
"""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from . import config
from .db import init_db
from .routers import admin, affiliations, chat, coauthorship
from .services import rag


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()  # idempotent backstop; Alembic owns real migrations in prod
    try:
        rag.warmup()  # pre-load embedding model to avoid cold-start latency
        print(f"RAG ready: {rag.count()} chunks indexed")
    except Exception as e:  # never let a RAG hiccup stop the server booting
        print(f"RAG warmup skipped: {type(e).__name__}: {e}")
    yield


def create_app() -> FastAPI:
    app = FastAPI(title="alex-loftus.com API", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(chat.router)
    app.include_router(coauthorship.router)
    app.include_router(affiliations.router)
    app.include_router(admin.router)
    return app


app = create_app()
