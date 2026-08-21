"""Fire-and-forget Slack heads-ups via an incoming webhook.

Runs as a FastAPI background task after the response is sent, so a slow or
down Slack can never block or fail the request that triggered it. With no
webhook configured (config.KLIST_SLACK_WEBHOOK unset) this is a no-op.
"""

from __future__ import annotations

import logging

import httpx

from .. import config

logger = logging.getLogger(__name__)


def klist_notify(text: str) -> None:
    """POST a one-line message to the klist Slack webhook; swallow all errors."""
    url = config.KLIST_SLACK_WEBHOOK  # read at call time so tests can patch it
    if not url:
        return
    try:
        httpx.post(url, json={"text": text}, timeout=5.0)
    except Exception:
        logger.warning("klist Slack notification failed", exc_info=True)
