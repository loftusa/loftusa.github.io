"""Reached-out store for the /houses page.

Unauthenticated, personal low-stakes feature. Keyed by the Craigslist listing
URL (stable across the daily data.js rebuild). The static /houses page on
alex-loftus.com calls these cross-origin (CORS already allows that origin).
"""

from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from ..db import get_db
from ..models import HouseReachedOut

router = APIRouter(prefix="/houses", tags=["houses"])


class ReachedOutIn(BaseModel):
    url: str
    title: str | None = None
    message: str | None = None
    channel: str | None = None  # email | text | listing


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
    body: ReachedOutIn, db: Session = Depends(get_db)
) -> HouseReachedOut:
    """Mark a listing reached-out (upsert by URL)."""
    if not body.url:
        raise HTTPException(status_code=422, detail="url required")
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
def delete_reached_out(url: str, db: Session = Depends(get_db)) -> dict:
    """Un-mark a listing (by ?url=...)."""
    row = db.get(HouseReachedOut, url)
    if row is None:
        raise HTTPException(status_code=404, detail="not found")
    db.delete(row)
    db.commit()
    return {"ok": True}
