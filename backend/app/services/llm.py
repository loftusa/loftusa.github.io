"""LLM access (Cerebras). Imported lazily so the rest of the app — and the test suite — never
needs the cerebras SDK at import time. This is also the single seam where P6's /translate etc.
will hang their own model calls.
"""
from __future__ import annotations

from functools import lru_cache

from .. import config


@lru_cache(maxsize=1)
def _client():
    from cerebras.cloud.sdk import Cerebras  # local import: heavy + prod-only

    return Cerebras(api_key=config.CEREBRAS_API_KEY)


@lru_cache(maxsize=1)
def system_message() -> dict:
    """The résumé chatbot's system message: prompt + résumé, cached after first read."""
    prompt = config.SYSTEM_PROMPT_PATH.read_text()
    resume = config.RESUME_PATH.read_text()
    return {"role": "system", "content": prompt + "\n\n" + resume}


def stream_chat(messages: list[dict]):
    """A streaming chat completion (generator of chunks)."""
    return _client().chat.completions.create(
        model=config.MODEL, messages=messages, stream=True
    )
