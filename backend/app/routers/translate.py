"""Translation endpoint (/translate). Uses the Anthropic API lazily so this module imports
without the anthropic SDK. Guarded like the other open POST endpoints — per-IP rate limit +
an input-size cap — so a request loop can't burn the Anthropic budget.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session

from .. import config
from ..db import get_db
from ..deps import client_ip
from ..services import rate_limit
from ..services.translate import stream_translation

router = APIRouter(tags=["translate"])


class TranslateRequest(BaseModel):
    text: str


@router.post("/translate")
def translate(req: TranslateRequest, request: Request, db: Session = Depends(get_db)):
    if not req.text.strip():
        raise HTTPException(status_code=422, detail="text must not be empty")
    if len(req.text) > config.MAX_TRANSLATE_CHARS:
        raise HTTPException(
            status_code=422,
            detail=f"text too long (max {config.MAX_TRANSLATE_CHARS} chars)",
        )
    # Scoped key: shares the rate_limit_hits table with /corrections but must not drain
    # (or be drained by) the corrections budget.
    limit, window = config.TRANSLATE_RATE
    if not rate_limit.rate_ok(
        db, f"translate:{client_ip(request)}", limit=limit, window=window
    ):
        raise HTTPException(
            status_code=429, detail="too many translations; try again later"
        )

    def translation_stream():
        try:
            for chunk in stream_translation(req.text):
                yield chunk
        except Exception as e:
            print(f"Translation API error: {type(e).__name__}: {e}")
            yield f"[Error: translation failed — {type(e).__name__}]"

    return StreamingResponse(translation_stream(), media_type="text/plain")
