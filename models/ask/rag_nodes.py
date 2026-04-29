"""
rag_nodes.py
============
Five fully-async LangGraph nodes + streaming infrastructure.

Blocking I/O strategy
---------------------
  - All blocking calls (OpenAI, ChromaDB) run in the default thread-pool
    executor so FastAPI's event loop is never blocked.
  - generate_text        → _in_executor       (returns one complete string)
  - generate_text_stream → _stream_in_executor (Queue bridge, token by token)

Streaming architecture
----------------------
  The sync generator produced by provider.generate_text_stream() cannot be
  passed to run_in_executor directly — that would exhaust the generator inside
  the thread and return nothing to the event loop.

  Instead, _stream_in_executor runs a producer function in the thread pool that
  pushes each token into an asyncio.Queue via loop.call_soon_threadsafe.
  The event loop is already waiting on the queue before the first token arrives.
  It wakes up the moment one token lands, yields it to the client immediately,
  then goes back to waiting for the next one.

                Thread pool                     Queue            Event loop
                ───────────                   ─────────          ──────────
  token1  ──call_soon_threadsafe──►  [token1]  ◄──await──  yield token1 → client
  token2  ──call_soon_threadsafe──►  [token2]  ◄──await──  yield token2 → client
  None    ──call_soon_threadsafe──►  [None]    ◄──await──  stop

Node catalogue
--------------
  1. retrieve          – embed question → query ChromaDB → raw_chunks
  2. grade_chunks      – LLM relevance filter (all chunks graded concurrently)
  3. answer            – NOT used as a graph node anymore.
                         Streaming is handled in the router via stream_answer().
  4. extract_links     – pure-Python URL deduplication (zero LLM calls)
  5. summarize_history – LLM compresses rolling summary + new Q&A (followup only)
                         Called after streaming completes so state.answer is full.

Public helpers consumed by rag_router.py
-----------------------------------------
  build_answer_prompt  – builds the LLM messages list from state
  stream_answer        – async generator: yields tokens via Queue bridge
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from functools import partial
from typing import Any, AsyncGenerator

from langchain_core.runnables import RunnableConfig

from .rag_deps  import RAGDeps
from .rag_state import RAGState

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Private helpers
# ─────────────────────────────────────────────────────────────────────────────

async def _in_executor(fn, *args) -> Any:
    """Run a blocking callable in the thread pool. Returns one value."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, partial(fn, *args))


def _get_deps(config: RunnableConfig) -> RAGDeps:
    """Extract RAGDeps from the LangGraph RunnableConfig."""
    deps: RAGDeps | None = config.get("configurable", {}).get("deps")
    if deps is None:
        raise RuntimeError(
            "RAGDeps not found in config. "
            "Pass deps via graph.ainvoke(state, config={'configurable': {'deps': deps}})"
        )
    return deps


def _msgs(*parts: tuple[str, str]) -> list[dict]:
    """Build an OpenAI-style messages list from (role, content) tuples."""
    return [{"role": r, "content": c} for r, c in parts]


def _parse_verdict(raw: str) -> bool:
    """Safely parse the grader's {relevant: true/false} JSON response."""
    try:
        clean = re.sub(r"```(?:json)?|```", "", raw).strip()
        return bool(json.loads(clean).get("relevant", False))
    except Exception:
        return "true" in raw.lower()


# ─────────────────────────────────────────────────────────────────────────────
# Streaming bridge
# ─────────────────────────────────────────────────────────────────────────────

async def _stream_in_executor(sync_gen) -> AsyncGenerator[str, None]:
    """
    Pipe a blocking sync generator into the async event loop token by token.

    The producer runs in the thread pool and pushes each token into an
    asyncio.Queue via loop.call_soon_threadsafe — the only safe way to touch
    the event loop from a thread.

    The event loop is already suspended on await queue.get() before the first
    token arrives. It wakes up the moment one token lands, yields it to the
    caller immediately, then suspends again waiting for the next one.

    A None sentinel signals end of stream. It is sent inside a finally block
    so it is always delivered even if the generator raises mid-stream.
    """
    queue: asyncio.Queue[str | None] = asyncio.Queue()
    loop  = asyncio.get_running_loop()

    def _producer() -> None:
        try:
            for token in sync_gen:
                loop.call_soon_threadsafe(queue.put_nowait, token)
        finally:
            # always deliver sentinel — even if generator raises
            loop.call_soon_threadsafe(queue.put_nowait, None)

    # start producer in background — no await, event loop is free immediately
    loop.run_in_executor(None, _producer)

    while True:
        token = await queue.get()
        if token is None:   # sentinel — generator exhausted
            break
        yield token


# ─────────────────────────────────────────────────────────────────────────────
# Prompts
# ─────────────────────────────────────────────────────────────────────────────

_GRADER_SYSTEM = (
    "You are a strict relevance grader.\n"
    "Given a USER QUESTION and a TEXT CHUNK, decide if the chunk is USEFUL "
    "to answer the question.\n"
    "Respond ONLY with a JSON object — no markdown, no explanation:\n"
    '  {"relevant": true}   — chunk is helpful\n'
    '  {"relevant": false}  — chunk is off-topic'
)

_ANSWER_SYSTEM = (
    "You are a knowledgeable news assistant.\n"
    "Answer the user's question clearly and concisely using ONLY the provided "
    "context chunks. Write in the same language as the question.\n"
    "If the context is insufficient, say so honestly — never hallucinate.\n"
    "Do NOT list source URLs in your answer; they are returned separately."
)

_SUMMARIZE_SYSTEM = (
    "You are a conversation memory manager.\n"
    "Produce a SHORT, dense summary (≤ 200 words) that captures everything "
    "a future assistant needs to continue the conversation intelligently.\n\n"
    "Rules:\n"
    "- Write in third-person neutral style.\n"
    "- Merge the EXISTING SUMMARY (if any) with the NEW EXCHANGE.\n"
    "- Preserve key facts, entities, decisions, and open questions.\n"
    "- Drop filler, pleasantries, and redundant phrasing.\n"
    "- Respond with ONLY the summary text — no preamble, no labels."
)


# ─────────────────────────────────────────────────────────────────────────────
# Public helpers  (consumed by rag_router for streaming)
# ─────────────────────────────────────────────────────────────────────────────

def build_answer_prompt(state: RAGState) -> list[dict] | None:
    """
    Build the LLM messages list from state.
    Returns None if there are no chunks so the caller can handle
    the empty-context case without making an LLM call.

    Prompt order (when available):
      1. Rolling summary        (followup only)
      2. Last verbatim exchange (followup only)
      3. Relevant chunks        (falls back to raw_chunks)
      4. Current question
    """
    chunks = state.relevant_chunks or state.raw_chunks
    if not chunks:
        return None

    context = "\n\n".join(f"[{i + 1}] {c['document']}" for i, c in enumerate(chunks))

    user_parts: list[str] = []

    if state.summary:
        user_parts.append(f"CONVERSATION SUMMARY SO FAR:\n{state.summary}")

    if state.last_exchange:
        user_parts.append(
            f"LAST EXCHANGE:\n"
            f"Q: {state.last_exchange['question']}\n"
            f"A: {state.last_exchange['answer']}"
        )

    user_parts.append(f"CONTEXT:\n{context}")
    user_parts.append(f"QUESTION:\n{state.question}")

    return _msgs(
        ("system", _ANSWER_SYSTEM),
        ("user",   "\n\n".join(user_parts)),
    )


async def stream_answer(
    messages: list[dict],
    deps: RAGDeps,
) -> AsyncGenerator[str, None]:
    """
    Stream the LLM answer token by token via the Queue bridge.
    Yields tokens as the thread pool produces them.
    """
    sync_gen = deps.generation_provider.generate_text_stream(messages)
    async for token in _stream_in_executor(sync_gen):
        yield token


# ─────────────────────────────────────────────────────────────────────────────
# Node 1 · retrieve
# ─────────────────────────────────────────────────────────────────────────────

async def retrieve(state: RAGState, config: RunnableConfig) -> RAGState:
    """
    1. Call provider.embed_text(question)  → float vector
    2. Query ChromaDB collection           → top-k closest chunks
    3. Write results to state.raw_chunks
    """
    deps = _get_deps(config)
    logger.info("[RETRIEVE] question=%r  top_k=%d", state.question, deps.top_k)

    embedding: list[float] | None = await _in_executor(
        deps.embed_provider.embed_text, state.question
    )

    if not embedding:
        logger.error("[RETRIEVE] embed_text returned None — check provider config.")
        state.raw_chunks = []
        return state

    results = deps.collection.query(
        query_embeddings=[embedding],
        n_results=deps.top_k,
        include=["documents", "metadatas", "distances"],
    )

    state.raw_chunks = [
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
    ]

    logger.info("[RETRIEVE] %d chunks fetched.", len(state.raw_chunks))
    return state


# ─────────────────────────────────────────────────────────────────────────────
# Node 2 · grade_chunks
# ─────────────────────────────────────────────────────────────────────────────

async def grade_chunks(state: RAGState, config: RunnableConfig) -> RAGState:
    """
    Grade every raw chunk for relevance concurrently.
    All grading calls are fired with asyncio.gather so latency is
    O(1 LLM call) regardless of chunk count.
    """
    deps = _get_deps(config)
    logger.info("[GRADE] grading %d chunks…", len(state.raw_chunks))

    async def _grade_one(chunk: dict) -> bool:
        messages = _msgs(
            ("system", _GRADER_SYSTEM),
            ("user",
             f"USER QUESTION:\n{state.question}\n\n"
             f"TEXT CHUNK:\n{chunk['document']}"),
        )
        raw: str | None = await _in_executor(deps.generation_provider.generate_text, messages)
        return _parse_verdict(raw or "")

    verdicts: list[bool] = await asyncio.gather(
        *[_grade_one(c) for c in state.raw_chunks]
    )
    print(state.raw_chunks)

    state.relevant_chunks = [
        c for c, ok in zip(state.raw_chunks, verdicts) if ok
    ]
    print("=============================="*10)
    print(state.relevant_chunks)
    print("=============================="*10)


    logger.info(
        "[GRADE] %d/%d chunks kept.",
        len(state.relevant_chunks), len(state.raw_chunks),
    )
    return state


# ─────────────────────────────────────────────────────────────────────────────
# Node 3 · extract_links
# ─────────────────────────────────────────────────────────────────────────────

async def extract_links(state: RAGState, config: RunnableConfig) -> RAGState:
    """
    Pure Python — zero LLM calls, zero I/O.
    Reads chunk metadata and builds a deduplicated list of source URLs.
    Called manually by the router after streaming completes.
    """
    chunks = state.relevant_chunks or state.raw_chunks
    seen:  set[str]   = set()
    links: list[dict] = []

    for chunk in chunks:
        meta  = chunk.get("metadata", {})
        url   = (meta.get("url")   or "").strip()
        title = (meta.get("title") or url).strip()

        if url and url not in seen:
            seen.add(url)
            links.append({"title": title, "url": url})

    state.source_links = links
    logger.info("[LINKS] %d unique source(s).", len(links))
    return state


# ─────────────────────────────────────────────────────────────────────────────
# Node 4 · summarize_history  (followup only)
# ─────────────────────────────────────────────────────────────────────────────

async def summarize_history(state: RAGState, config: RunnableConfig) -> RAGState:
    """
    Compress (existing summary + new Q&A) into a rolling summary ≤ 200 words.
    Called manually by the router after streaming completes so state.answer
    is guaranteed to be the full answer string.

    Result written to state.updated_summary.
    Router persists it to MongoDB via session_store.update_session().
    """
    deps = _get_deps(config)
    logger.info("[SUMMARY] compressing history…")

    existing     = state.summary or "No prior conversation."
    new_exchange = f"Q: {state.question}\nA: {state.answer}"

    messages = _msgs(
        ("system", _SUMMARIZE_SYSTEM),
        ("user",
         f"EXISTING SUMMARY:\n{existing}\n\n"
         f"NEW EXCHANGE:\n{new_exchange}\n\n"
         "Produce the updated summary now."),
    )

    raw: str | None = await _in_executor(deps.generation_provider.generate_text, messages)
    state.updated_summary = (raw or "").strip() or existing

    logger.info("[SUMMARY] new summary: %d words.", len(state.updated_summary.split()))
    return state