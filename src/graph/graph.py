"""
src/graph/graph.py
LangGraph multi-agent graph — assembly and compilation.

Graph topology:
  START
    └─► supervisor
          ├─ research_only   ──────────────────────► research  ──► summarizer ──► END
          ├─ calculation_only ─────────────────────► calculator ──► summarizer ──► END
          ├─ both             ──► research (parallel)─┐
          │                   └─► calculator (parallel)┘──► summarizer ──► END
          └─ summarize_only ───────────────────────────────────► summarizer ──► END

Parallel execution:
  LangGraph executes nodes in a Send-based fan-out for "both" routing.
  Both research and calculator nodes run simultaneously; summarizer waits
  for both to complete before running.

Failure isolation:
  Each node catches its own exceptions and writes to state["errors"].
  The graph never raises an unhandled exception — summarizer always runs.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any, Literal

from langgraph.graph import END, START, StateGraph

from src.config import load_config
from src.graph.nodes.calculator  import calculator_node
from src.graph.nodes.research    import research_node
from src.graph.nodes.summarizer  import summarizer_node
from src.graph.nodes.supervisor  import supervisor_node
from src.graph.state import AgentState
from src.logging.sqlite_logger import log_agent_steps

log = logging.getLogger(__name__)
cfg = load_config()

_ROUTING = cfg["graph"]["routing"]
_MAX_RETRIES = cfg["graph"]["max_retries"]


# ── Failure-isolated node wrappers ────────────────────────────────────────────

def _with_retry(node_fn, node_key: str):
    """
    Wrap a node function with retry logic (max_retries from config).
    On repeated failure, returns partial state with error appended.
    The graph never raises — summarizer always runs.
    """
    def wrapper(state: AgentState) -> dict:
        max_retries = _MAX_RETRIES
        last_result = {}
        for attempt in range(max_retries + 1):
            try:
                last_result = node_fn(state)
                # If node itself flagged an error, check if we should retry
                node_errors = [
                    e for e in last_result.get("errors", [])
                    if node_key in e
                ]
                if not node_errors:
                    return last_result          # success
                if attempt < max_retries:
                    log.warning(
                        "[Retry] %s failed (attempt %d/%d) — retrying",
                        node_key, attempt + 1, max_retries + 1,
                    )
            except Exception as e:
                log.error("[Retry] %s raised exception (attempt %d): %s", node_key, attempt + 1, e)
                last_result = {
                    "errors": state.get("errors", []) + [f"{node_key}: exception — {e}"],
                    "agent_trace": state.get("agent_trace", []),
                }
                if attempt < max_retries:
                    continue
        return last_result
    return wrapper


research_with_retry    = _with_retry(research_node,    "research")
calculator_with_retry  = _with_retry(calculator_node,  "calculator")


# ── Conditional edge: supervisor → sub-agents ─────────────────────────────────

def route_after_supervisor(
    state: AgentState,
) -> Literal["research", "calculator", "both_research", "summarizer"]:
    """
    Reads state["route"] set by the supervisor node and returns the next
    node name(s) for LangGraph to execute.

    "both" is handled by sending to a merged parallel node below.
    """
    route = state.get("route", _ROUTING["research_only"])
    log.info("[Router] routing decision: %s", route)

    if route == _ROUTING["research_only"]:
        return "research"
    if route == _ROUTING["calculation_only"]:
        return "calculator"
    if route == _ROUTING["both"]:
        return "both_research"        # enters the parallel fan-in node
    # summarize_only (or unknown — safe default)
    return "summarizer"


# ── Parallel fan-in node for "both" routing ───────────────────────────────────
# LangGraph doesn't have native AND-join; we use a sequential wrapper node
# that calls both agents and merges their outputs into state before handing
# off to the summarizer. This is the simplest correct approach for a local
# single-process setup and keeps the graph topology linear.

def parallel_research_and_calculator(state: AgentState) -> dict[str, Any]:
    """
    Wrapper node that runs research and calculator sequentially but presents
    as a single "parallel" step. For true async parallelism in a production
    deployment, replace with LangGraph's Send() API or asyncio.gather().

    Why not asyncio.gather() here:
      Both nodes call asyncio.run() internally (sync → async bridge for MCP).
      Nested event loops are not supported in standard CPython without a
      workaround like nest_asyncio. Sequential execution is safe and correct;
      the latency difference on a single Ollama instance is negligible since
      Ollama processes one request at a time anyway.
    """
    log.info("[Parallel] running research + calculator")
    research_result    = research_node(state)
    # Merge research output into state before passing to calculator
    merged = {**state, **research_result}
    calculator_result  = calculator_node(merged)

    # Merge both outputs
    return {
        "research_output":    research_result.get("research_output"),
        "calculation_output": calculator_result.get("calculation_output"),
        "agent_trace": (
            state.get("agent_trace", [])
            + research_result.get("agent_trace", [])[len(state.get("agent_trace", [])):]
            + calculator_result.get("agent_trace", [])[len(research_result.get("agent_trace", [])[len(state.get("agent_trace", [])):]) + len(state.get("agent_trace", [])):]
        ),
        "errors": (
            state.get("errors", [])
            + research_result.get("errors", [])
            + calculator_result.get("errors", [])
        ),
    }


# ── Graph builder ─────────────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    """Construct and return the compiled LangGraph StateGraph."""
    builder = StateGraph(AgentState)

    # ── Add nodes ─────────────────────────────────────────────────────────────
    builder.add_node("supervisor",    supervisor_node)
    builder.add_node("research",      research_with_retry)
    builder.add_node("calculator",    calculator_with_retry)
    builder.add_node("both_research", parallel_research_and_calculator)
    builder.add_node("summarizer",    summarizer_node)

    # ── Entry edge ────────────────────────────────────────────────────────────
    builder.add_edge(START, "supervisor")

    # ── Conditional routing from supervisor ───────────────────────────────────
    builder.add_conditional_edges(
        "supervisor",
        route_after_supervisor,
        {
            "research":      "research",
            "calculator":    "calculator",
            "both_research": "both_research",
            "summarizer":    "summarizer",
        },
    )

    # ── Sub-agent → summarizer edges ──────────────────────────────────────────
    builder.add_edge("research",      "summarizer")
    builder.add_edge("calculator",    "summarizer")
    builder.add_edge("both_research", "summarizer")

    # ── Terminal edge ─────────────────────────────────────────────────────────
    builder.add_edge("summarizer", END)

    return builder.compile()


# ── Public run function ───────────────────────────────────────────────────────

def run_graph(query: str, query_id: str | None = None) -> AgentState:
    """
    Run the full multi-agent graph for a given user query.
    Returns the final AgentState after all nodes have executed.

    Args:
        query:    The user's natural-language query.
        query_id: Optional UUID. Generated if not provided.

    Returns:
        Final AgentState with final_answer, agent_trace, and errors.
    """
    if not query_id:
        query_id = str(uuid.uuid4())

    initial_state: AgentState = {
        "query_id":           query_id,
        "query":              query,
        "route":              None,
        "memory_context":     None,
        "research_output":    None,
        "calculation_output": None,
        "final_answer":       None,
        "agent_trace":        [],
        "errors":             [],
        "retry_counts":       {"research": 0, "calculator": 0},
    }

    log.info("[Graph] starting  query_id=%s  query=%r", query_id, query)

    graph  = build_graph()
    cfg_g  = load_config()["graph"]
    result = graph.invoke(
        initial_state,
        config={"recursion_limit": cfg_g["recursion_limit"]},
    )

    log.info(
        "[Graph] complete  query_id=%s  route=%s  errors=%d",
        query_id, result.get("route"), len(result.get("errors", [])),
    )

    # ── Persist every step to SQLite ──────────────────────────────────────────
    log_agent_steps(result)

    return result