from pydantic import BaseModel, Field
from typing import List, Optional


class RAGState(BaseModel):
    """
    Shared mutable state that flows through every graph node.

    Fresh-ask graph   → summary / last_exchange are always empty / None.
    Followup graph    → summary carries the rolling compressed history;
                        last_exchange carries the previous verbatim Q&A.
    """

    # ── inputs ───────────────────────────────────────────────────────────────
    question: str = ""

    # ── memory (followup graph only) ─────────────────────────────────────────
    summary: str = ""
    """Rolling compressed summary — fed to the answer node as context."""

    last_exchange: Optional[dict] = None
    """{"question": str, "answer": str} — verbatim previous turn."""

    # ── retrieval ────────────────────────────────────────────────────────────
    raw_chunks: List[dict] = Field(default_factory=list)
    relevant_chunks: List[dict] = Field(default_factory=list)

    # ── generation ───────────────────────────────────────────────────────────
    answer: str = ""

    # ── sources ──────────────────────────────────────────────────────────────
    source_links: List[dict] = Field(default_factory=list)
    """[{"title": str, "url": str}]"""

    # ── written by summarize_history; caller persists to MongoDB ─────────────
    updated_summary: str = ""