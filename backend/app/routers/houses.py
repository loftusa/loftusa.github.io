"""Reached-out store for the /houses page.

Unauthenticated but hardened, per Alex's call: the data is low-stakes (which
listings he contacted), so instead of auth we validate hard and rate-limit —
only https craigslist.org listing URLs are accepted, fields are length-capped,
and POST/DELETE share a per-IP budget (scoped key, so it can't drain other
endpoints' budgets). Keyed by listing URL (stable across the daily data.js
rebuild). The static /houses page calls these cross-origin.
"""

from __future__ import annotations

from datetime import datetime
from urllib.parse import urlparse

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field, field_validator
from sqlalchemy.orm import Session

from .. import config
from ..db import get_db
from ..deps import client_ip
from ..models import HouseReachedOut
from ..services import rate_limit

router = APIRouter(prefix="/houses", tags=["houses"])


def _check_rate(request: Request, db: Session) -> None:
    limit, window = config.HOUSES_RATE
    if not rate_limit.rate_ok(
        db, f"houses:{client_ip(request)}", limit=limit, window=window
    ):
        raise HTTPException(status_code=429, detail="slow down; try again later")


class ReachedOutIn(BaseModel):
    url: str = Field(max_length=600)
    title: str | None = Field(None, max_length=300)
    message: str | None = Field(None, max_length=4000)
    channel: str | None = Field(None, max_length=16)  # email | text | listing

    @field_validator("url")
    @classmethod
    def craigslist_only(cls, v: str) -> str:
        p = urlparse(v)
        host = p.hostname or ""
        if p.scheme != "https" or not (
            host == "craigslist.org" or host.endswith(".craigslist.org")
        ):
            raise ValueError("url must be an https craigslist.org listing")
        return v


class ReachedOutOut(BaseModel):
    url: str
    title: str | None
    message: str | None
    channel: str | None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


@router.get("/reached-out", response_model=list[ReachedOutOut])
def list_reached_out(db: Session = Depends(get_db)) -> list[HouseReachedOut]:
    """Every listing that's been reached out about (newest first)."""
    return db.query(HouseReachedOut).order_by(HouseReachedOut.updated_at.desc()).all()


@router.post("/reached-out", response_model=ReachedOutOut)
def upsert_reached_out(
    body: ReachedOutIn, request: Request, db: Session = Depends(get_db)
) -> HouseReachedOut:
    """Mark a listing reached-out (upsert by URL)."""
    _check_rate(request, db)
    row = db.get(HouseReachedOut, body.url)
    if row is None:
        row = HouseReachedOut(
            url=body.url,
            title=body.title,
            message=body.message,
            channel=body.channel,
        )
        db.add(row)
    else:
        if body.title is not None:
            row.title = body.title
        if body.message is not None:
            row.message = body.message
        if body.channel is not None:
            row.channel = body.channel
    db.commit()
    db.refresh(row)
    return row


@router.delete("/reached-out")
def delete_reached_out(
    url: str, request: Request, db: Session = Depends(get_db)
) -> dict:
    """Un-mark a listing (by ?url=...)."""
    _check_rate(request, db)
    row = db.get(HouseReachedOut, url)
    if row is None:
        raise HTTPException(status_code=404, detail="not found")
    db.delete(row)
    db.commit()
    return {"ok": True}
