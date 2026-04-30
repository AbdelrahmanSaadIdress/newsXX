"""
article_nodes.py
================
All pipeline logic for the /article/digest workflow.

Blocking-I/O strategy
----------------------
  Every call that touches the network, disk, or a GPU model is run in
  FastAPI's default thread-pool executor via run_in_threadpool() or
  asyncio.get_running_loop().run_in_executor(), so the event loop is
  never blocked.

  Streaming uses the same asyncio.Queue bridge that rag_nodes.py uses:
  the sync OpenAI generator runs in a thread and pushes tokens via
  loop.call_soon_threadsafe; the event loop wakes up for each token and
  yields it to the client immediately.

Public API (consumed by article_router.py)
------------------------------------------
  lookup_or_scrape_and_analyse  – Phase 0
  retrieve_bullets               – Phase 1
  build_digest_prompt            – prompt builder
  stream_digest                  – async token generator (Phase 2)
  extract_source_links           – deduplicated source list
"""

from __future__ import annotations

import asyncio
import logging
from functools import partial
from typing import AsyncGenerator, Any

from fastapi.concurrency import run_in_threadpool

from models.analyzingANDtranslating.analyze_Trans_deps import A_TDeps
from models.ask.rag_deps import RAGDeps
from .article_state import ArticleState

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Private async helpers
# ─────────────────────────────────────────────────────────────────────────────

async def _in_executor(fn, *args) -> Any:
    """Run a blocking callable in the default thread pool."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, partial(fn, *args))


async def _stream_in_executor(sync_gen) -> AsyncGenerator[str, None]:
    """
    Pipe a blocking sync generator into the async event loop token by token.

    Identical to the Queue bridge in rag_nodes.py — see that module for a
    detailed explanation of the threading model.
    """
    queue: asyncio.Queue[str | None] = asyncio.Queue()
    loop = asyncio.get_running_loop()

    def _producer() -> None:
        try:
            for token in sync_gen:
                loop.call_soon_threadsafe(queue.put_nowait, token)
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, None)

    loop.run_in_executor(None, _producer)

    while True:
        token = await queue.get()
        if token is None:
            break
        yield token


# ─────────────────────────────────────────────────────────────────────────────
# Prompts
# ─────────────────────────────────────────────────────────────────────────────

_DIGEST_SYSTEM = (
    "You are an expert investigative journalist and senior editor.\n"
    "You are given a set of key facts (summary bullets) extracted from a news article, "
    "together with relevant supporting chunks retrieved from a wider news knowledge base.\n\n"
    "Your task:\n"
    "  Write a single, polished, professionally structured news article that:\n"
    "  1. Opens with a compelling lead paragraph that captures the most important fact.\n"
    "  2. Develops each key point in its own paragraph, weaving in supporting evidence "
    "     from the retrieved context chunks where relevant.\n"
    "  3. Closes with a concise concluding paragraph that contextualises the story.\n\n"
    "Style rules:\n"
    "  - Write in the same language as the key facts.\n"
    "  - Use clear, authoritative, active-voice prose.\n"
    "  - Do NOT number the paragraphs or use bullet points in the output.\n"
    "  - Do NOT invent statistics or quotes — rely solely on the provided material.\n"
    "  - Do NOT list source URLs in the body of the article."
)


# ─────────────────────────────────────────────────────────────────────────────
# Phase 0 — Lookup / scrape / analyse
# ─────────────────────────────────────────────────────────────────────────────

async def lookup_or_scrape_and_analyse(
    url: str,
    rag_deps: RAGDeps,
    at_deps: A_TDeps,
) -> ArticleState:
    """
    1. Check the analysis MongoDB collection for the URL.
       Hit  → populate ArticleState directly from the stored document.
       Miss → scrape the page, run generate_analysis() in the thread pool,
              persist the result to Mongo, then populate ArticleState.

    Both paths return an ArticleState with all analysis fields filled in.
    """
    state = ArticleState(url=url)

    # ── check Mongo ───────────────────────────────────────────────────────────
    existing = await run_in_threadpool(
        _find_analysis_doc, at_deps, url
    )

    if existing:
        logger.info("[DIGEST] Cache HIT for url=%r", url)
        _populate_state_from_doc(state, existing)
        return state

    # ── cache miss: scrape ────────────────────────────────────────────────────
    logger.info("[DIGEST] Cache MISS — scraping url=%r", url)
    raw_content = await run_in_threadpool(_scrape_url, url)

    if not raw_content:
        raise ValueError(f"Scraper returned no content for URL: {url}")

    state.raw_content = raw_content

    # ── analyse (blocking GPU/CPU call) ──────────────────────────────────────
    logger.info("[DIGEST] Running generate_analysis for url=%r", url)
    news_details = await run_in_threadpool(at_deps.generate_analysis, raw_content)
    print("====="*20)
    print(news_details)
    print("====="*20)

    if news_details is None:
        raise ValueError("Analysis model failed to produce a valid result.")

    # ── persist to Mongo ──────────────────────────────────────────────────────
    await run_in_threadpool(
        _persist_analysis, at_deps, url, news_details
    )

    # ── populate state ────────────────────────────────────────────────────────
    state.story_title    = news_details.story_title
    state.story_keywords = news_details.story_keywords
    state.story_summary  = news_details.story_summary
    state.story_category = news_details.story_category
    state.story_entities = [e.model_dump() for e in news_details.story_entities]

    return state


# ── sync helpers (run inside thread pool) ─────────────────────────────────────

def _find_analysis_doc(at_deps: A_TDeps, url: str) -> dict | None:
    """Synchronous Mongo lookup — runs in the thread pool."""
    from pymongo import MongoClient
    from helpers.Config import get_settings

    settings = get_settings()
    client   = MongoClient(settings.MONGO_URL)
    try:
        col = client[settings.MONGO_DB][settings.MONGO_COL_ANALYSIS]
        return col.find_one({"url": url}, {"_id": 0})
    finally:
        client.close()


def _scrape_url(url: str) -> str:
    """
    Scrape a single article URL and return its plain-text content.
    Reuses PageScraper's HTTP + parsing logic without saving anything.
    """
    from pyquery import PyQuery as pq

    page      = pq(url=url)
    container = page(".wysiwyg--all-content")

    parts = []
    for element in container.children().items():
        text = element.text().strip()
        if text:
            parts.append(text)

    return "\n".join(parts)


def _persist_analysis(at_deps: A_TDeps, url: str, news_details) -> None:
    """
    Persist a freshly generated analysis document to Mongo.
    Mirrors _analyse_and_store() in PageScraping.py but with
    current_day=None and description=None (single-URL deep-dive context).
    """
    from pymongo import MongoClient
    from helpers.Config import get_settings

    settings = get_settings()
    client   = MongoClient(settings.MONGO_URL)
    try:
        col = client[settings.MONGO_DB][settings.MONGO_COL_ANALYSIS]
        doc = {
            "url":            url,
            "title":          news_details.story_title,
            "description":    None,
            "current_day":    None,
            "story_title":    news_details.story_title,
            "story_keywords": news_details.story_keywords,
            "story_summary":  news_details.story_summary,
            "story_category": news_details.story_category,
            "story_entities": [e.model_dump() for e in news_details.story_entities],
        }
        try:
            col.insert_one(doc)
            logger.info("[DIGEST] Analysis persisted for url=%r", url)
        except Exception as exc:
            # Duplicate key or write error — not fatal, log and move on
            logger.warning("[DIGEST] Mongo insert skipped for %r: %s", url, exc)
    finally:
        client.close()


def _populate_state_from_doc(state: ArticleState, doc: dict) -> None:
    """Fill ArticleState fields from a Mongo analysis document."""
    state.story_title    = doc.get("story_title", "")
    state.story_keywords = doc.get("story_keywords", [])
    state.story_summary  = doc.get("story_summary", [])
    state.story_category = doc.get("story_category", "")
    state.story_entities = doc.get("story_entities", [])


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1 — Per-bullet retrieval
# ─────────────────────────────────────────────────────────────────────────────

async def retrieve_bullets(
    state: ArticleState,
    rag_deps: RAGDeps,
    top_k: int | None = None,
    exclude_url: str | None = None,
) -> ArticleState:
    """
    For every summary bullet, embed the bullet text and query ChromaDB for
    the most relevant chunks.  Chunks whose ``url`` metadata matches the
    source article are excluded so we only surface *related* articles.

    All embedding calls are fired concurrently with asyncio.gather so total
    latency is O(1 embedding call) regardless of bullet count.

    Result written to state.bullets_with_chunks.
    """
    k           = top_k or rag_deps.top_k
    exclude_url = exclude_url or state.url

    logger.info(
        "[DIGEST] Retrieving chunks for %d bullets (top_k=%d, exclude=%r)",
        len(state.story_summary), k, exclude_url,
    )

    async def _retrieve_one(bullet: str) -> dict:
        embedding: list[float] | None = await _in_executor(
            rag_deps.embed_provider.embed_text, bullet
        )
        if not embedding:
            logger.warning("[DIGEST] embed_text returned None for bullet=%r", bullet[:60])
            return {"bullet": bullet, "chunks": []}

        results = await _in_executor(
            lambda emb: rag_deps.collection.query(
                query_embeddings=[emb],
                n_results=k,
                include=["documents", "metadatas", "distances"],
            ),
            embedding,
        )

        # Unpack and exclude chunks from the source article itself
        raw_chunks = [
            {
                "document": doc,
                "metadata": meta,
                "distance": round(dist, 4),
            }
            for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            )
            if meta.get("url", "") != exclude_url
        ]

        return {"bullet": bullet, "chunks": raw_chunks}

    # Run all bullets concurrently
    state.bullets_with_chunks = await asyncio.gather(
        *[_retrieve_one(b) for b in state.story_summary]
    )

    total_chunks = sum(len(b["chunks"]) for b in state.bullets_with_chunks)
    logger.info("[DIGEST] Retrieved %d total supporting chunks.", total_chunks)
    return state


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2 — Prompt builder + streaming generation
# ─────────────────────────────────────────────────────────────────────────────

def build_digest_prompt(state: ArticleState) -> list[dict]:
    """
    Build the LLM messages list from state.bullets_with_chunks.

    Structure
    ---------
      ARTICLE OVERVIEW
        Title / Category / Keywords

      KEY FACTS WITH SUPPORTING CONTEXT
        Bullet 1:  <text>
          Supporting context:
            [1] <chunk>
            [2] <chunk>
        Bullet 2: …

      TASK
        Write the article now.
    """
    # ── overview block ────────────────────────────────────────────────────────
    overview_lines = [
        "ARTICLE OVERVIEW",
        f"  Title    : {state.story_title}",
        f"  Category : {state.story_category}",
        f"  Keywords : {', '.join(state.story_keywords)}",
        "",
    ]

    # ── bullets + chunks block ────────────────────────────────────────────────
    facts_lines: list[str] = ["KEY FACTS WITH SUPPORTING CONTEXT"]

    for idx, item in enumerate(state.bullets_with_chunks, 1):
        bullet = item["bullet"]
        chunks = item["chunks"]

        facts_lines.append(f"\n[Fact {idx}] {bullet}")

        if chunks:
            facts_lines.append("  Supporting context from related articles:")
            for c_idx, chunk in enumerate(chunks, 1):
                # Truncate very long chunks so the prompt stays reasonable
                doc = chunk["document"][:600]
                facts_lines.append(f"    [{c_idx}] {doc}")
        else:
            facts_lines.append("  (No supporting context found in knowledge base.)")

    facts_lines.append("")

    task_lines = [
        "TASK",
        "Write the complete article now, following the style rules provided.",
    ]

    user_content = "\n".join(overview_lines + facts_lines + task_lines)

    return [
        {"role": "system", "content": _DIGEST_SYSTEM},
        {"role": "user",   "content": user_content},
    ]


async def stream_digest(
    messages: list[dict],
    rag_deps: RAGDeps,
) -> AsyncGenerator[str, None]:
    """
    Stream the generated article token by token via the Queue bridge.
    Mirrors stream_answer() in rag_nodes.py exactly.
    """
    sync_gen = rag_deps.generation_provider.generate_text_stream(messages)
    async for token in _stream_in_executor(sync_gen):
        yield token


# ─────────────────────────────────────────────────────────────────────────────
# Source link extractor
# ─────────────────────────────────────────────────────────────────────────────

def extract_source_links(state: ArticleState) -> list[dict]:
    """
    Collect a deduplicated list of source URLs from all retrieved chunks,
    excluding the original article URL.
    """
    seen:  set[str]   = set()
    links: list[dict] = []

    for item in state.bullets_with_chunks:
        for chunk in item["chunks"]:
            meta  = chunk.get("metadata", {})
            url   = (meta.get("url")   or "").strip()
            title = (meta.get("title") or url).strip()

            if url and url not in seen and url != state.url:
                seen.add(url)
                links.append({"title": title, "url": url})

    logger.info("[DIGEST] %d unique source link(s) collected.", len(links))
    return links