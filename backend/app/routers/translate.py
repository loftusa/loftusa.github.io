"""Translation endpoint (/translate). Uses the Anthropic API lazily so this module imports
without the anthropic SDK.
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from ..services.translate import stream_translation

router = APIRouter(tags=["translate"])


class TranslateRequest(BaseModel):
    text: str


@router.post("/translate")
def translate(req: TranslateRequest):
    if not req.text.strip():
        raise HTTPException(status_code=422, detail="text must not be empty")

    def translation_stream():
        try:
            for chunk in stream_translation(req.text):
                yield chunk
        except Exception as e:
            print(f"Translation API error: {type(e).__name__}: {e}")
            yield f"[Error: translation failed — {type(e).__name__}]"

    return StreamingResponse(translation_stream(), media_type="text/plain")
