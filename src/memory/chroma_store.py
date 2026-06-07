"""
src/memory/chroma_store.py
ChromaDB agent memory — stub for Phase 3.
Fully implemented in Phase 4.

The supervisor and summarizer nodes import from here at runtime.
Returning safe no-op values keeps the graph functional in Phase 3.
"""

from __future__ import annotations

from src.graph.state import AgentState


def retrieve_similar_traces(query: str) -> str:
    """
    Phase 3 stub: returns empty string (no memory context injected).
    Phase 4: queries ChromaDB for semantically similar past traces.
    """
    return ""


def persist_trace(state: AgentState) -> None:
    """
    Phase 3 stub: no-op.
    Phase 4: embeds and stores the completed reasoning trace in ChromaDB.
    """
    pass