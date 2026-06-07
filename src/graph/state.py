"""
src/graph/state.py
Shared state schema for the multi-agent LangGraph graph.

This TypedDict is the single object that flows through every node.
Every agent reads from and writes to this state — it is the only
communication channel between nodes.

Design notes:
  - All fields are Optional so nodes can safely read keys set by other nodes.
  - `errors` is a list so multiple agents can each append failures without
    one overwriting another's error.
  - `agent_trace` accumulates one entry per node execution for observability.
  - `memory_context` is injected by the supervisor from ChromaDB before
    routing — sub-agents never write to it.
"""

from __future__ import annotations

from typing import Any, Optional
from typing_extensions import TypedDict


class AgentStep(TypedDict):
    """One entry in the agent_trace list — logged per node execution."""
    agent_name:  str
    tool_called: Optional[str]
    input:       Any
    output:      Any
    latency_ms:  float
    success:     bool


class AgentState(TypedDict):
    # ── Identity ──────────────────────────────────────────────────────────────
    query_id: str                          # UUID assigned at graph entry
    query:    str                          # raw user query

    # ── Routing ───────────────────────────────────────────────────────────────
    route: Optional[str]                   # set by supervisor:
                                           #   "research_only" | "calculation_only"
                                           #   | "both" | "summarize_only"

    # ── Memory (injected by supervisor before routing) ────────────────────────
    memory_context: Optional[str]          # serialised past traces from ChromaDB

    # ── Sub-agent outputs ─────────────────────────────────────────────────────
    research_output:    Optional[str]      # set by research agent
    calculation_output: Optional[str]      # set by calculator agent

    # ── Final answer ──────────────────────────────────────────────────────────
    final_answer: Optional[str]            # set by summarizer agent

    # ── Observability ─────────────────────────────────────────────────────────
    agent_trace: list[AgentStep]           # one entry per node execution
    errors:      list[str]                 # each failed node appends here

    # ── Retry tracking ────────────────────────────────────────────────────────
    retry_counts: dict[str, int]           # {"research": 0, "calculator": 0}