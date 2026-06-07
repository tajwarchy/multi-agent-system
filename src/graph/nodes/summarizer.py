"""
src/graph/nodes/summarizer.py
Summarizer agent node.

Reads research_output and/or calculation_output from shared state and
synthesises a single coherent final answer for the user.

Also triggers:
  - ChromaDB persistence of the completed reasoning trace (Phase 4 hook).
  - SQLite logging is handled at the graph level (Phase 4).
"""

from __future__ import annotations

import logging
import time
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from src.config import load_config
from src.graph.state import AgentState, AgentStep
from src.llm import get_llm

log = logging.getLogger(__name__)
cfg = load_config()

_SYSTEM_PROMPT = cfg["summarizer_agent"]["system_prompt"]


def _persist_trace(state: AgentState) -> None:
    """
    Persist the completed reasoning trace to ChromaDB.
    Fully implemented in Phase 4. No-op until then.
    """
    try:
        from src.memory.chroma_store import persist_trace
        persist_trace(state)
    except Exception:
        pass   # non-fatal — observability only


def _build_context(state: AgentState) -> str:
    """Assemble all available agent outputs into a single context block."""
    parts = [f"User query: {state['query']}"]

    if state.get("memory_context"):
        parts.append(f"\n--- Relevant past context ---\n{state['memory_context']}")

    if state.get("research_output"):
        parts.append(f"\n--- Research findings ---\n{state['research_output']}")

    if state.get("calculation_output"):
        parts.append(f"\n--- Calculation results ---\n{state['calculation_output']}")

    if state.get("errors"):
        parts.append(
            f"\n--- Partial failures (incorporate gracefully) ---\n"
            + "\n".join(state["errors"])
        )

    return "\n".join(parts)


def summarizer_node(state: AgentState) -> dict[str, Any]:
    """
    LangGraph node — synthesises the final answer from all available context.
    Returns partial state dict.
    """
    t0 = time.monotonic()
    log.info("[Summarizer] building final answer")

    context = _build_context(state)
    success = True

    final_answer = "I was unable to generate a final answer due to an internal error."

    try:
        llm      = get_llm()
        messages = [
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(content=context),
        ]
        response     = llm.invoke(messages)
        final_answer = response.content.strip()
        log.info("[Summarizer] final answer (%d chars)", len(final_answer))
    except Exception as e:
        log.error("[Summarizer] LLM call failed: %s", e)
        # Graceful degradation — return whatever raw context we have
        final_answer = (
            "Summarisation failed. Here is the raw output:\n\n" + context
        )
        success = False

    latency_ms = (time.monotonic() - t0) * 1000

    step: AgentStep = {
        "agent_name":  "summarizer",
        "tool_called": None,
        "input":       context[:300],   # truncate for trace readability
        "output":      final_answer[:500],
        "latency_ms":  round(latency_ms, 2),
        "success":     success,
    }

    errors = list(state.get("errors", []))
    if not success:
        errors.append("summarizer_node: LLM call failed")

    # ── Persist trace to ChromaDB (Phase 4 hook) ──────────────────────────────
    updated_state = {
        **state,
        "final_answer": final_answer,
        "agent_trace":  state.get("agent_trace", []) + [step],
        "errors":       errors,
    }
    _persist_trace(updated_state)  # type: ignore[arg-type]

    return {
        "final_answer": final_answer,
        "agent_trace":  state.get("agent_trace", []) + [step],
        "errors":       errors,
    }