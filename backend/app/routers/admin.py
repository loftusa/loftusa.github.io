"""Admin / log-export endpoints (bearer-gated). The chat-log download reconstructs the original
chat_logs.jsonl line shape from the DB so sync_logs.py / analyze_logs.py keep working unchanged.
"""
from __future__ import annotations

import json

from fastapi import APIRouter, Depends, Header
from fastapi.responses import Response
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..db import get_db
from ..deps import require_bearer
from ..models import ChatLog

router = APIRouter(tags=["admin"])


@router.get("/logs/download")
def download_logs(
    authorization: str | None = Header(None), db: Session = Depends(get_db)
):
    """Bearer-protected JSONL export of the chat logs (same field shape as the old file)."""
    require_bearer(authorization)
    rows = db.scalars(select(ChatLog).order_by(ChatLog.id)).all()
    lines = "".join(
        json.dumps(
            {
                "user_id": r.user_id,
                "user_message": r.user_message,
                "bot_response": r.bot_response,
                "token_latency_ms": r.token_latency_ms,
                "timestamp": r.ts,
                "message_count": r.message_count,
                "token_count": r.token_count,
                "conversation_cost_usd": r.conversation_cost_usd,
                "rag_chunks_count": r.rag_chunks_count,
                "rag_sources": r.rag_sources or [],
            }
        )
        + "\n"
        for r in rows
    )
    return Response(content=lines, media_type="text/plain")
