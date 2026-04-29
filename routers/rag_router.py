"""
rag_router.py
=============
RAG ask endpoints — both return true token-level StreamingResponse.

  POST /api/v1/ask           – fresh question  (no session memory)
  POST /api/v1/ask/followup  – follow-up question (session_id required)

Both endpoints follow the same three-phase pattern:

  Phase 1 — retrieval  (via retrieval_graph)
    retrieve → grade_chunks
    Runs fully before streaming starts so the prompt can be built.

  Phase 2 — streaming  (via Queue bridge)
    Tokens flow from OpenAI thread → asyncio.Queue → client in real time.
    Collected into state.answer as they arrive.

  Phase 3 — post-stream  (after last token)
    extract_links         → state.source_links         (both flows)
    summarize_history     → state.updated_summary      (followup only)
    store.update_session  → persist to MongoDB         (followup only)
    [DONE] SSE frame      → source_links + session_id  (both flows)

Wire format  (text/event-stream)
---------------------------------
  Each token:     data: <token text>\n\n
  Final frame:    data: [DONE] {"source_links": [...], "session_id": "..."}\n\n
"""

from __future__ import annotations

import json
import logging
from typing import AsyncGenerator, Optional

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from models.ask.rag_deps   import RAGDeps
from models.ask.rag_graph  import retrieval_graph
from models.ask.rag_nodes  import (
    build_answer_prompt,
    extract_links,
    stream_answer,
    summarize_history,
)
from models.ask.rag_state  import RAGState
from models.ask.session_store import MongoSessionStore, SessionNotFoundError

logger = logging.getLogger(__name__)

_NO_CONTEXT_MSG = "I could not find any relevant articles to answer your question."


# ─────────────────────────────────────────────────────────────────────────────
# Schemas
# ─────────────────────────────────────────────────────────────────────────────

class AskRequest(BaseModel):
    question: str


class FollowupRequest(BaseModel):
    session_id: str
    question: str


# ─────────────────────────────────────────────────────────────────────────────
# Dependency helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_rag_deps(request: Request) -> RAGDeps:
    return request.app.rag_deps


def get_session_store(request: Request) -> MongoSessionStore:
    return request.app.session_store


# ─────────────────────────────────────────────────────────────────────────────
# SSE helpers
# ─────────────────────────────────────────────────────────────────────────────

def _sse_token(token: str) -> str:
    """Format a single token as an SSE data line."""
    return f"data: {token}\n\n"


def _sse_done(source_links: list[dict], session_id: str | None = None) -> str:
    """Format the final metadata frame that closes the stream."""
    payload: dict = {"source_links": source_links}
    if session_id:
        payload["session_id"] = session_id
    return f"data: [DONE] {json.dumps(payload, ensure_ascii=False)}\n\n"


# ─────────────────────────────────────────────────────────────────────────────
# Shared retrieval phase
# ─────────────────────────────────────────────────────────────────────────────

async def _run_retrieval(state: RAGState, config: dict) -> RAGState:
    """Run retrieve → grade_chunks via the retrieval graph."""
    return await retrieval_graph.ainvoke(state, config=config)


# ─────────────────────────────────────────────────────────────────────────────
# Stream generators
# ─────────────────────────────────────────────────────────────────────────────

async def _fresh_stream(
    question: str,
    deps: RAGDeps,
    config: dict,
) -> AsyncGenerator[str, None]:
    """
    Phase 1 — retrieve + grade_chunks
    Phase 2 — stream answer tokens to client, collect into state.answer
    Phase 3 — extract_links → [DONE] frame
    """
    # Phase 1
    state = await _run_retrieval(RAGState(question=question), config)
    state = RAGState(**state)
    
    # Phase 2
    messages = build_answer_prompt(state)

    if messages is None:
        yield _sse_token(_NO_CONTEXT_MSG)
        state.answer = _NO_CONTEXT_MSG
    else:
        collected: list[str] = []
        async for token in stream_answer(messages, deps):
            collected.append(token)
            yield _sse_token(token)
        state.answer = "".join(collected)

    # Phase 3
    state = await extract_links(state, config)
    yield _sse_done(state.source_links)


async def _followup_stream(
    question: str,
    session: dict,
    deps: RAGDeps,
    config: dict,
    store: MongoSessionStore,
    session_id: str,
) -> AsyncGenerator[str, None]:
    """
    Phase 1 — retrieve + grade_chunks  (with session memory loaded into state)
    Phase 2 — stream answer tokens to client, collect into state.answer
    Phase 3 — extract_links → summarize_history → persist to MongoDB → [DONE] frame
    """
    # Phase 1
    state = await _run_retrieval(
        RAGState(
            question=question,
            summary=session.get("summary", ""),
            last_exchange=session.get("last_exchange"),
        ),
        config,
    )
    state = RAGState(**state)
    # Phase 2
    messages = build_answer_prompt(state)

    if messages is None:
        yield _sse_token(_NO_CONTEXT_MSG)
        state.answer = _NO_CONTEXT_MSG
    else:
        collected: list[str] = []
        async for token in stream_answer(messages, deps):
            collected.append(token)
            yield _sse_token(token)
        state.answer = "".join(collected)

    # Phase 3 — state.answer is now complete, safe to summarize
    state = await extract_links(state, config)
    state = await summarize_history(state, config)

    await store.update_session(
        session_id=session_id,
        question=question,
        answer=state.answer,
        source_links=state.source_links,
        updated_summary=state.updated_summary,
    )

    yield _sse_done(state.source_links, session_id=session_id)


# ─────────────────────────────────────────────────────────────────────────────
# Router
# ─────────────────────────────────────────────────────────────────────────────

rag_router = APIRouter(prefix="/api/v1", tags=["RAG"])


@rag_router.post("/ask")
async def ask(
    body: AskRequest,
    deps: RAGDeps = Depends(get_rag_deps),
):
    """
    Answer a standalone question with true token-level streaming.
    No session is created; nothing is persisted.
    """
    config = {"configurable": {"deps": deps}}
    return StreamingResponse(
        _fresh_stream(body.question, deps, config),
        media_type="text/event-stream",
        headers={"X-Accel-Buffering": "no"},
    )


@rag_router.post("/ask/followup")
async def ask_followup(
    body: FollowupRequest,
    deps: RAGDeps = Depends(get_rag_deps),
    store: MongoSessionStore = Depends(get_session_store),
):
    """
    Answer a follow-up question within an existing session with true streaming.

    Phase 1: retrieve + grade_chunks (memory loaded from MongoDB into state)
    Phase 2: stream answer tokens to client as they arrive
    Phase 3: extract_links + summarize_history + persist to MongoDB
    """
    try:
        session = await store.get_session(body.session_id)
    except SessionNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))

    config = {"configurable": {"deps": deps}}
    return StreamingResponse(
        _followup_stream(
            question=body.question,
            session=session,
            deps=deps,
            config=config,
            store=store,
            session_id=body.session_id,
        ),
        media_type="text/event-stream",
        headers={"X-Accel-Buffering": "no"},
    )