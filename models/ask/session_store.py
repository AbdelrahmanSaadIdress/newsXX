"""
session_store.py
================
Async MongoDB-backed session store.

Collection schema  (chat_sessions)
-----------------------------------
{
    "_id":           ObjectId,
    "session_id":    str   (uuid4, indexed unique),
    "summary":       str   (rolling compressed summary — fed to the LLM),
    "last_exchange": {     (verbatim last turn — fed to the LLM as recent context)
        "question": str,
        "answer":   str
    } | None,
    "history": [           (full audit trail — never trimmed)
        {
            "question":   str,
            "answer":     str,
            "sources":    [{"title": str, "url": str}],
            "turn_index": int,
            "asked_at":   datetime
        },
        ...
    ],
    "turn_count":  int,
    "created_at":  datetime,
    "updated_at":  datetime   ← TTL index lives here (set via ensure_indexes)
}
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Optional

from motor.motor_asyncio import AsyncIOMotorClient

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Data shapes
# ─────────────────────────────────────────────────────────────────────────────

class SessionNotFoundError(Exception):
    """Raised when a session_id does not exist in the store."""


# ─────────────────────────────────────────────────────────────────────────────
# Store
# ─────────────────────────────────────────────────────────────────────────────

class MongoSessionStore:
    """
    Async session store backed by MongoDB.

    One instance is created at application startup and shared across
    all requests via FastAPI dependency injection.

    Parameters
    ----------
    mongo_url : str
        Full MongoDB connection string.
    db_name : str
        Database name (reuses your existing MONGO_DB setting).
    collection_name : str
        Collection for chat sessions. Defaults to "chat_sessions".
    ttl_days : int
        Inactive sessions older than this many days are auto-deleted
        by MongoDB's TTL index on `updated_at`.  Default: 30 days.
    """

    def __init__(
        self,
        mongo_url: str,
        db_name: str,
        collection_name: str = "chat_sessions",
        ttl_days: int = 30,
    ):
        self._client = AsyncIOMotorClient(mongo_url)
        self._col = self._client[db_name][collection_name]
        self._ttl_seconds = ttl_days * 24 * 60 * 60
        logger.info(
            "MongoSessionStore initialised — db=%s  col=%s  ttl=%dd",
            db_name, collection_name, ttl_days,
        )

    # ── lifecycle ─────────────────────────────────────────────────────────────

    async def ensure_indexes(self) -> None:
        """
        Create indexes once at startup.
        Safe to call multiple times — MongoDB is idempotent for existing indexes.

            1. Unique index on session_id  → fast O(1) lookups.
            2. TTL index on updated_at     → MongoDB auto-deletes stale sessions.
        """
        await self._col.create_index("session_id", unique=True)
        await self._col.create_index(
            "updated_at",
            expireAfterSeconds=self._ttl_seconds,
        )
        logger.info("MongoSessionStore indexes ensured.")

    async def close(self) -> None:
        self._client.close()

    # ── public API ────────────────────────────────────────────────────────────

    async def create_session(self) -> str:
        """
        Insert a blank session document and return the new session_id.
        """
        session_id = str(uuid.uuid4())
        now = _utcnow()

        await self._col.insert_one({
            "session_id":    session_id,
            "summary":       "",
            "last_exchange": None,
            "history":       [],
            "turn_count":    0,
            "created_at":    now,
            "updated_at":    now,
        })

        logger.info("Session created: %s", session_id)
        return session_id

    async def get_session(self, session_id: str) -> dict:
        """
        Fetch a session document.

        Returns the raw MongoDB document (without _id).
        Raises SessionNotFoundError if the session does not exist.
        """
        doc = await self._col.find_one(
            {"session_id": session_id},
            {"_id": 0},
        )
        if doc is None:
            raise SessionNotFoundError(
                f"Session '{session_id}' not found. "
                "It may have expired or never existed."
            )
        return doc

    async def update_session(
        self,
        session_id: str,
        question: str,
        answer: str,
        source_links: list[dict],
        updated_summary: str,
    ) -> None:
        """
        Atomically update a session after one completed turn:

            - $set  summary        ← new rolling summary from summarize_history node
            - $set  last_exchange  ← verbatim current Q&A (used by next turn's answer node)
            - $inc  turn_count
            - $push a full record into history[]
            - $set  updated_at     ← refreshes the TTL clock

        Parameters
        ----------
        session_id      : str   — the session to update
        question        : str   — the question asked this turn
        answer          : str   — the answer generated this turn
        source_links    : list  — [{title, url}] from extract_links node
        updated_summary : str   — the compressed summary from summarize_history node
        """
        now = _utcnow()

        result = await self._col.update_one(
            {"session_id": session_id},
            {
                "$set": {
                    "summary":       updated_summary,
                    "last_exchange": {"question": question, "answer": answer},
                    "updated_at":    now,
                },
                "$inc": {"turn_count": 1},
                "$push": {
                    "history": {
                        "question":   question,
                        "answer":     answer,
                        "sources":    source_links,
                        "asked_at":   now,
                    }
                },
            },
        )

        if result.matched_count == 0:
            raise SessionNotFoundError(
                f"Session '{session_id}' not found during update."
            )

        logger.info(
            "Session updated: %s  (summary_len=%d)",
            session_id, len(updated_summary),
        )

    async def get_history(self, session_id: str) -> list[dict]:
        """
        Return the full turn-by-turn history for a session.
        Useful for a 'show me my past conversation' UI endpoint.
        """
        doc = await self.get_session(session_id)
        return doc.get("history", [])

    async def delete_session(self, session_id: str) -> None:
        """Explicitly delete a session (e.g. user presses 'clear chat')."""
        await self._col.delete_one({"session_id": session_id})
        logger.info("Session deleted: %s", session_id)


# ─────────────────────────────────────────────────────────────────────────────
# helpers
# ─────────────────────────────────────────────────────────────────────────────

def _utcnow() -> datetime:
    return datetime.now(timezone.utc)