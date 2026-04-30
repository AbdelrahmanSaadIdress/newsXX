"""
article_router.py
=================
Two-route deep-dive pipeline that turns any article URL into a freshly
written, context-enriched piece — optionally translated into English or French.

Routes
------
POST /api/v1/article/digest
    ──────────────────────────────────────────────────────────────────────
    Full pipeline, fully streamed:

    Phase 0  — Lookup
                Check analysis collection in MongoDB.
                Hit  → grab analysis straight from Mongo.
                Miss → scrape URL with PageScraper, run generate_analysis()
                        (blocking; offloaded to thread pool), persist to Mongo.

    Phase 1  — Multi-bullet retrieval
                For every summary bullet, embed + query ChromaDB to find
                supporting chunks from *other* articles.

    Phase 2  — Streaming generation
                Build a rich prompt from all bullets + their supporting
                chunks and stream a professionally written article back to
                the client via SSE, exactly like rag_router.

    Wire format (text/event-stream)
    ─────────────────────────────────
    First frame  : data: [META] { story_keywords, story_category,
                                    story_summary, story_title,
                                    story_entities }\\n\\n
    Token frames : data: <token>\\n\\n
    Final frame  : data: [DONE] { "source_links": [...] }\\n\\n

POST /api/v1/article/translate
    ──────────────────────────────────────────────────────────────────────
    Translates the generated article produced by /digest.

    Request body  : { "article": "<full text>", "lang": "en" | "fr" }
    Response body : { "translated_title": "...", "translated_content": "..." }

    Blocking call offloaded to thread pool.
"""

from __future__ import annotations

import json
import logging
from typing import AsyncGenerator

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from models.analyzingANDtranslating.analyze_Trans_deps import A_TDeps
from models.ask.rag_deps import RAGDeps
from models.analyzingANDtranslating.article_state import ArticleState
from models.analyzingANDtranslating.article_nodes import (
    lookup_or_scrape_and_analyse,
    retrieve_bullets,
    build_digest_prompt,
    stream_digest,
    extract_source_links,
)

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Schemas
# ─────────────────────────────────────────────────────────────────────────────

class DigestRequest(BaseModel):
    url: str = Field(..., description="Full URL of the article to deep-dive.")


class TranslateRequest(BaseModel):
    article: str = Field(..., description="Full generated article text to translate.")
    lang: str    = Field(..., pattern="^(en|fr)$",
                        description="Target language: 'en' for English, 'fr' for French.")


class TranslateResponse(BaseModel):
    translated_title:   str
    translated_content: str


# ─────────────────────────────────────────────────────────────────────────────
# Dependency helpers
# ─────────────────────────────────────────────────────────────────────────────

def _get_rag_deps(request: Request) -> RAGDeps:
    return request.app.rag_deps


def _get_at_deps(request: Request) -> A_TDeps:
    return request.app.A_T_deps


# ─────────────────────────────────────────────────────────────────────────────
# SSE helpers
# ─────────────────────────────────────────────────────────────────────────────

def _sse_meta(state: ArticleState) -> str:
    payload = {
        "story_title":    state.story_title,
        "story_keywords": state.story_keywords,
        "story_category": state.story_category,
        "story_summary":  state.story_summary,
        "story_entities": state.story_entities,
    }
    return f"data: [META] {json.dumps(payload, ensure_ascii=False)}\n\n"


def _sse_token(token: str) -> str:
    return f"data: {token}\n\n"


def _sse_done(source_links: list[dict]) -> str:
    payload = {"source_links": source_links}
    return f"data: [DONE] {json.dumps(payload, ensure_ascii=False)}\n\n"


# ─────────────────────────────────────────────────────────────────────────────
# Stream generator
# ─────────────────────────────────────────────────────────────────────────────

async def _digest_stream(
    url: str,
    rag_deps: RAGDeps,
    at_deps: A_TDeps,
) -> AsyncGenerator[str, None]:
    """
    Full three-phase pipeline streamed back as SSE.

    Phase 0  — lookup / scrape / analyse  (blocking → thread pool)
    Phase 1  — multi-bullet ChromaDB retrieval  (async, concurrent)
    Phase 2  — streaming generation via Queue bridge
    """
    # ── Phase 0: lookup or scrape + analyse ──────────────────────────────────
    try:
        state = await lookup_or_scrape_and_analyse(url, rag_deps, at_deps)
    except Exception as exc:
        logger.exception("[DIGEST] Phase-0 failure for url=%r", url)
        yield _sse_token(f"[ERROR] Could not process article: {exc}")
        return

    if not state.story_summary:
        yield _sse_token("[ERROR] Analysis produced no summary bullets — cannot continue.")
        return

    # ── Send meta frame immediately so the client can render metadata ─────────
    yield _sse_meta(state)

    # ── Phase 1: per-bullet retrieval ─────────────────────────────────────────
    try:
        state = await retrieve_bullets(state, rag_deps)
    except Exception as exc:
        logger.exception("[DIGEST] Phase-1 retrieval failure")
        yield _sse_token(f"[ERROR] Retrieval failed: {exc}")
        return

    # ── Phase 2: streaming generation ─────────────────────────────────────────
    messages = build_digest_prompt(state)

    collected: list[str] = []
    async for token in stream_digest(messages, rag_deps):
        collected.append(token)
        yield _sse_token(token)

    state.generated_article = "".join(collected)

    # ── Final frame: source links ─────────────────────────────────────────────
    source_links = extract_source_links(state)
    yield _sse_done(source_links)


# ─────────────────────────────────────────────────────────────────────────────
# Router
# ─────────────────────────────────────────────────────────────────────────────

article_router = APIRouter(prefix="/api/v1/article", tags=["Article Deep-Dive"])


@article_router.post(
    "/digest",
    summary="Generate a deep-dive article with streaming",
    response_description=(
        "SSE stream: [META] frame → token frames → [DONE] frame"
    ),
)
async def digest(
    body:     DigestRequest,
    rag_deps: RAGDeps = Depends(_get_rag_deps),
    at_deps:  A_TDeps = Depends(_get_at_deps),
):
    """
    Turn any article URL into a freshly written, context-enriched deep-dive.

    Stream format
    -------------
    - ``data: [META] {...}``   — analysis metadata (first frame)
    - ``data: <token>``        — generated article tokens
    - ``data: [DONE] {...}``   — source links (last frame)
    """
    return StreamingResponse(
        _digest_stream(body.url, rag_deps, at_deps),
        media_type="text/event-stream",
        headers={"X-Accel-Buffering": "no"},
    )


@article_router.post(
    "/translate",
    response_model=TranslateResponse,
    summary="Translate a generated article into English or French",
)
async def translate(
    body:    TranslateRequest,
    at_deps: A_TDeps = Depends(_get_at_deps),
):
    """
    Translate the article text produced by ``/digest``.

    Pass the full ``generated_article`` text collected on the client side
    (everything streamed between the ``[META]`` and ``[DONE]`` frames)
    together with the desired target language (``en`` or ``fr``).
    """
    if body.lang == "en":
        translation_fn = at_deps.generate_english_translation
    else:
        translation_fn = at_deps.generate_french_translation

    print("ggggggggggggggggggggggggggggggggggggggggggg")
    print(body.article)
    print("ggggggggggggggggggggggggggggggggggggggggggg")

    result = await run_in_threadpool(translation_fn, body.article)

    if result is None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Translation model failed to produce a valid output. Please retry.",
        )

    return TranslateResponse(
        translated_title=result.translated_title,
        translated_content=result.translated_content,
    )