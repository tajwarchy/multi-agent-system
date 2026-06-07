"""
src/graph/nodes/supervisor.py
Supervisor agent node.

Responsibilities:
  1. Retrieve semantically similar past traces from ChromaDB (memory stub —
     wired fully in Phase 4; returns "" until then).
  2. Classify the user query and output a routing decision.
  3. Write `route` and `memory_context` into shared state.

Routing decisions (from config/config.yaml → graph.routing):
  - research_only      → Research agent only
  - calculation_only   → Calculator agent only
  - both               → Research + Calculator in parallel
  - summarize_only     → Skip sub-agents; Summarizer works from memory context
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from src.config import load_config
from src.graph.state import AgentState, AgentStep
from src.llm import get_llm

log = logging.getLogger(__name__)
cfg = load_config()

_ROUTING_LABELS = set(cfg["graph"]["routing"].values())
_SYSTEM_PROMPT  = cfg["supervisor"]["system_prompt"]


def _get_memory_context(query: str) -> str:
    """
    Retrieve past traces from ChromaDB relevant to this query.
    Fully implemented in Phase 4. Returns empty string until then.
    """
    try:
        from src.memory.chroma_store import retrieve_similar_traces
        return retrieve_similar_traces(query)
    except Exception:
        return ""


def _build_prompt(query: str, memory_context: str) -> list:
    messages = [SystemMessage(content=_SYSTEM_PROMPT)]
    if memory_context:
        messages.append(
            HumanMessage(content=(
                f"Relevant past reasoning traces (use as context if helpful):\n"
                f"{memory_context}\n\n"
                f"---\nUser query: {query}"
            ))
        )
    else:
        messages.append(HumanMessage(content=f"User query: {query}"))
    return messages


def supervisor_node(state: AgentState) -> dict[str, Any]:
    """
    LangGraph node — classifies the query and sets `route` in shared state.
    Returns a partial state dict (LangGraph merges it with existing state).
    """
    t0    = time.monotonic()
    query = state["query"]
    log.info("[Supervisor] query=%r", query)

    # ── 1. Retrieve memory context ────────────────────────────────────────────
    memory_context = _get_memory_context(query)

    # ── 2. Call LLM to classify ───────────────────────────────────────────────
    llm      = get_llm()
    messages = _build_prompt(query, memory_context)

    route    = "research_only"   # safe default
    success  = True
    llm_raw  = ""

    try:
        response = llm.invoke(messages)
        llm_raw  = response.content.strip().lower()

        # Extract the routing label — tolerate minor LLM verbosity
        matched = None
        for label in _ROUTING_LABELS:
            if label in llm_raw:
                matched = label
                break

        if matched:
            route = matched
            log.info("[Supervisor] route=%s", route)
        else:
            log.warning(
                "[Supervisor] LLM returned unexpected output %r — defaulting to 'research_only'",
                llm_raw,
            )

    except Exception as e:
        log.error("[Supervisor] LLM call failed: %s — defaulting to 'research_only'", e)
        success = False

    latency_ms = (time.monotonic() - t0) * 1000

    step: AgentStep = {
        "agent_name":  "supervisor",
        "tool_called": None,
        "input":       query,
        "output":      route,
        "latency_ms":  round(latency_ms, 2),
        "success":     success,
    }

    return {
        "route":          route,
        "memory_context": memory_context,
        "agent_trace":    state.get("agent_trace", []) + [step],
        "errors":         state.get("errors", []),
        "retry_counts":   state.get("retry_counts", {"research": 0, "calculator": 0}),
    }