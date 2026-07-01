"""Résumé chatbot endpoints (/chat, /reset, /health). Logic mirrors the old chat_api.py, but the
conversation store, cost cap, and chat log are now durable (SQLite). LLM + RAG are loaded lazily
so this module imports without cerebras/chromadb.
"""
from __future__ import annotations

import time
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session

from .. import config
from ..db import SessionLocal, get_db
from ..models import ChatLog
from ..services import conversations as conv
from ..services import llm, rag

router = APIRouter(tags=["chat"])


class ChatRequest(BaseModel):
    user_id: str | None = None
    message: str


class ResetRequest(BaseModel):
    user_id: str


def _build_messages(history: list[dict], rag_text: str | None) -> list[dict]:
    msgs = [llm.system_message()]
    if rag_text:
        msgs.append({"role": "system", "content": rag_text})
    return msgs + history


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).replace(tzinfo=None).isoformat()


@router.post("/chat")
def chat_stream(
    request: ChatRequest, logging: bool = True, db: Session = Depends(get_db)
):
    user_id = conv.get_or_create(db, request.user_id)

    # reject if this conversation already blew its cost budget (durable across restarts now)
    if conv.over_budget(db, user_id, config.MAX_COST_PER_CONVERSATION):
        raise HTTPException(
            status_code=429,
            detail="Conversation budget exceeded. Please start a new conversation.",
        )

    conv.trim_turns(db, user_id, config.MAX_TURNS)
    conv.append_message(db, user_id, "user", request.message)

    rag_chunks = rag.retrieve(request.message)
    rag_text = rag.format_context(rag_chunks)
    messages_for_llm = _build_messages(conv.messages(db, user_id), rag_text)

    def token_stream():
        try:
            completion = llm.stream_chat(messages_for_llm)
        except Exception as e:
            print(f"LLM API error: {type(e).__name__}: {e}")
            yield f"[Error: LLM API call failed — {type(e).__name__}]"
            return

        bot_response = ""
        completion_tokens = -1
        prompt_tokens = -1
        start = time.time()
        chunk_count = 0
        try:
            for chunk in completion:
                delta = chunk.choices[0].delta
                if delta and delta.content:
                    bot_response += delta.content
                    chunk_count += 1
                    yield delta.content
                if hasattr(chunk, "usage") and chunk.usage:
                    if chunk.usage.completion_tokens:
                        completion_tokens = chunk.usage.completion_tokens
                    if chunk.usage.prompt_tokens:
                        prompt_tokens = chunk.usage.prompt_tokens
        except Exception as e:
            print(f"LLM streaming error: {type(e).__name__}: {e}")
            yield f"\n[Error: streaming interrupted — {type(e).__name__}]"
            return

        if completion_tokens == -1:
            completion_tokens = chunk_count
        latency = (time.time() - start) * 1000

        # post-stream persistence uses a FRESH session: the request's get_db session may already be
        # torn down by the time this generator finishes streaming.
        wdb = SessionLocal()
        try:
            if prompt_tokens > 0 and completion_tokens > 0:
                turn_cost = (
                    prompt_tokens * config.INPUT_COST_PER_TOKEN
                    + completion_tokens * config.OUTPUT_COST_PER_TOKEN
                )
                conv.add_cost(wdb, user_id, turn_cost)
            conv.append_message(wdb, user_id, "assistant", bot_response)
            cost_now = conv.get_cost(wdb, user_id)
            token_latency_ms = (
                latency / completion_tokens if completion_tokens not in (0, -1) else -1
            )
            if logging:
                wdb.add(
                    ChatLog(
                        user_id=user_id,
                        user_message=request.message,
                        bot_response=bot_response,
                        token_latency_ms=token_latency_ms,
                        ts=_utcnow_iso(),
                        message_count=len(conv.messages(wdb, user_id)),
                        token_count=completion_tokens,
                        conversation_cost_usd=round(cost_now, 6),
                        rag_chunks_count=len(rag_chunks),
                        rag_sources=[c["source"] for c in rag_chunks],
                    )
                )
                wdb.commit()
        finally:
            wdb.close()

    return StreamingResponse(token_stream(), media_type="text/plain")


@router.post("/reset")
def reset_conversation(request: ResetRequest, db: Session = Depends(get_db)) -> dict:
    conv.reset(db, request.user_id)
    return {"status": "ok"}


@router.get("/health")
def health_check(db: Session = Depends(get_db)) -> dict:
    try:
        rag_chunks = rag.count()
    except Exception:
        rag_chunks = None  # RAG not loaded (e.g. in tests) — health is still OK
    return {
        "status": "ok",
        "api_key": bool(config.CEREBRAS_API_KEY),
        "rag_chunks": rag_chunks,
        "timestamp": _utcnow_iso(),
    }
