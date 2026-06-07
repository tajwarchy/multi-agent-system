"""
src/graph/nodes/research.py
Research agent node.

Calls DuckDuckGo search and Wikipedia lookup via MCP, then uses the LLM
to synthesise a structured research summary from the raw tool results.

Failure isolation:
  - Tool errors (network_error, page_not_found) are tolerated — the agent
    continues with whatever results it has.
  - If both tools fail, research_output is set to a descriptive error string
    so the supervisor can detect and handle it downstream.
  - Retries are managed by the supervisor (retry_counts in shared state).
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from src.config import load_config
from src.graph.state import AgentState, AgentStep
from src.llm import get_llm
from src.mcp_client import MCPClient

log = logging.getLogger(__name__)
cfg = load_config()

_SYSTEM_PROMPT = cfg["research_agent"]["system_prompt"]
_MAX_RETRIES   = cfg["graph"]["max_retries"]


async def _call_tools(query: str) -> dict[str, Any]:
    """Call both research tools via MCP and return their raw results."""
    async with MCPClient() as client:
        search_result = await client.call_tool(
            "duckduckgo_search", {"query": query, "max_results": 5}
        )
        wiki_result = await client.call_tool(
            "wikipedia_lookup", {"topic": query, "sentences": 5}
        )
    return {"search": search_result, "wikipedia": wiki_result}


def _format_tool_results(raw: dict[str, Any]) -> str:
    """Format raw MCP tool results into a readable string for the LLM."""
    parts = []

    search = raw.get("search", {})
    if "error" not in search and search.get("results"):
        parts.append("=== Web Search Results ===")
        for r in search["results"][:5]:
            parts.append(f"• {r['title']}\n  {r['snippet']}\n  URL: {r['url']}")
    elif "error" in search:
        parts.append(f"[Web search unavailable: {search['error']}]")

    wiki = raw.get("wikipedia", {})
    if "error" not in wiki:
        parts.append(f"\n=== Wikipedia: {wiki.get('title', '')} ===")
        parts.append(wiki.get("summary", ""))
    elif "error" in wiki:
        parts.append(f"[Wikipedia unavailable: {wiki['error']}]")

    return "\n".join(parts) if parts else "No research data available."


def research_node(state: AgentState) -> dict[str, Any]:
    """
    LangGraph node — runs research tools via MCP, synthesises a summary.
    Returns partial state dict.
    """
    t0    = time.monotonic()
    query = state["query"]
    log.info("[Research] starting  query=%r", query)

    tool_results_str = ""
    success          = True
    tool_name        = "duckduckgo_search + wikipedia_lookup"

    # ── 1. Call tools via MCP (async → sync bridge) ───────────────────────────
    try:
        raw              = asyncio.run(_call_tools(query))
        tool_results_str = _format_tool_results(raw)
        log.info("[Research] tools returned %d chars", len(tool_results_str))
    except Exception as e:
        log.error("[Research] MCP tool call failed: %s", e)
        tool_results_str = f"Tool call failed: {e}"
        success          = False

    # ── 2. LLM synthesis ──────────────────────────────────────────────────────
    research_output = tool_results_str   # fallback: return raw if LLM fails

    try:
        llm      = get_llm()
        messages = [
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(content=(
                f"User query: {query}\n\n"
                f"Raw research data:\n{tool_results_str}\n\n"
                "Synthesise a concise, factual research summary."
            )),
        ]
        response        = llm.invoke(messages)
        research_output = response.content.strip()
        log.info("[Research] synthesis complete (%d chars)", len(research_output))
    except Exception as e:
        log.error("[Research] LLM synthesis failed: %s", e)
        success = False

    latency_ms = (time.monotonic() - t0) * 1000

    step: AgentStep = {
        "agent_name":  "research",
        "tool_called": tool_name,
        "input":       query,
        "output":      research_output[:500],   # truncate for trace readability
        "latency_ms":  round(latency_ms, 2),
        "success":     success,
    }

    errors = list(state.get("errors", []))
    if not success:
        errors.append(f"research_node failed: partial output returned")

    return {
        "research_output": research_output,
        "agent_trace":     state.get("agent_trace", []) + [step],
        "errors":          errors,
    }