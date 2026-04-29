"""
session_router.py
=================
Session management endpoints.

    POST   /api/v1/sessions                      – create session
    DELETE /api/v1/sessions/{session_id}         – delete session
    GET    /api/v1/sessions/{session_id}/history – full turn history
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, status, Request
from pydantic import BaseModel

from models.ask.session_store import MongoSessionStore, SessionNotFoundError

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Schemas
# ─────────────────────────────────────────────────────────────────────────────

class SessionCreatedResponse(BaseModel):
    session_id: str


class HistoryResponse(BaseModel):
    session_id: str
    history: list[dict]


# ─────────────────────────────────────────────────────────────────────────────
# Dependency helper  (pulls shared store off app.state)
# ─────────────────────────────────────────────────────────────────────────────

def get_session_store(request:Request) -> MongoSessionStore:
    return request.app.session_store


# ─────────────────────────────────────────────────────────────────────────────
# Router
# ─────────────────────────────────────────────────────────────────────────────

session_router = APIRouter(prefix="/api/v1/sessions", tags=["Sessions"])


@session_router.post("",response_model=SessionCreatedResponse,status_code=status.HTTP_201_CREATED)
async def create_session(store: MongoSessionStore = Depends(get_session_store)):
    """Create a new conversation session and return its ID."""
    session_id = await store.create_session()
    return SessionCreatedResponse(session_id=session_id)


@session_router.delete("/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_session(
    session_id: str,
    store: MongoSessionStore = Depends(get_session_store),
):
    """Delete a session (e.g. user presses 'clear chat')."""
    try:
        await store.delete_session(session_id)
    except SessionNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))


@session_router.get("/{session_id}/history", response_model=HistoryResponse)
async def get_history(
    session_id: str,
    store: MongoSessionStore = Depends(get_session_store),
):
    """Return the full turn-by-turn history for a session."""
    try:
        history = await store.get_history(session_id)
    except SessionNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))
    return HistoryResponse(session_id=session_id, history=history)
