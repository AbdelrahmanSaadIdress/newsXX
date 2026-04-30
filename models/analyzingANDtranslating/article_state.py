"""
article_state.py
================
Shared state that flows through the article deep-dive workflow.
"""

from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field


class ArticleState(BaseModel):
    # ── input ────────────────────────────────────────────────────────────────
    url: str = ""

    # ── analysis (from Mongo or freshly generated) ───────────────────────────
    story_title:    str        = ""
    story_keywords: list[str]  = Field(default_factory=list)
    story_summary:  list[str]  = Field(default_factory=list)   # bullet points
    story_category: str        = ""
    story_entities: list[dict] = Field(default_factory=list)

    # ── retrieval — one list of chunks per summary bullet ────────────────────
    # bullets_with_chunks[i] = {"bullet": str, "chunks": [{"document": str, "metadata": dict}]}
    bullets_with_chunks: list[dict] = Field(default_factory=list)

    # ── generation ───────────────────────────────────────────────────────────
    generated_article: str = ""   # full streamed output collected here

    # ── scraped raw content (populated when the URL is not in Mongo) ─────────
    raw_content: str = ""