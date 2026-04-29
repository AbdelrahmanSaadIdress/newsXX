"""
rag_graph.py
============
One compiled LangGraph StateGraph shared by both fresh and followup flows.

  retrieval_graph – retrieve → grade_chunks

Both fresh and followup endpoints use this graph for the retrieval phase only.
Everything after (answer streaming, extract_links, summarize_history) is handled
manually in the router so the answer can be streamed token by token to the client
before summarize_history runs — which requires a complete state.answer string.

Flow in router
--------------
  1. retrieval_graph.ainvoke()          → state.raw_chunks, state.relevant_chunks
  2. stream_answer() via Queue bridge   → tokens flow to client, collected into state.answer
  3. extract_links(state)               → state.source_links
  4. summarize_history(state)           → state.updated_summary  (followup only)
  5. store.update_session(...)          → persist to MongoDB      (followup only)
"""

from __future__ import annotations

from langgraph.graph import StateGraph, END

from .rag_nodes import grade_chunks, retrieve
from .rag_state import RAGState


def _build_retrieval_graph() -> StateGraph:
    graph = StateGraph(RAGState)

    graph.add_node("retrieve",     retrieve)
    graph.add_node("grade_chunks", grade_chunks)

    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve",     "grade_chunks")
    graph.add_edge("grade_chunks", END)

    return graph.compile()


# ── compiled singleton ────────────────────────────────────────────────────────

retrieval_graph = _build_retrieval_graph()