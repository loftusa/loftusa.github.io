"""Anthropic translation service. Imported lazily so the rest of the app — and the test suite —
never needs the anthropic SDK at import time.
"""
from __future__ import annotations

from functools import lru_cache

from .. import config

SYSTEM = """You are a translator between English and Spanish.

Detect the language of the user's text:
- If English, translate to natural conversational Spanish.
- If Spanish, translate to natural conversational English.

Output ONLY the translation. No quotes, no preamble, no explanation, no language labels."""


@lru_cache(maxsize=1)
def _client():
    import anthropic  # local import: heavy + prod-only

    return anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)


def stream_translation(text: str):
    """A streaming translation (generator of text chunks)."""
    with _client().messages.stream(
        model=config.TRANSLATE_MODEL,
        max_tokens=1024,
        system=SYSTEM,
        messages=[{"role": "user", "content": text}],
    ) as s:
        for chunk in s.text_stream:
            yield chunk
