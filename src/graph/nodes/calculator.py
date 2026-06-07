"""
src/graph/nodes/calculator.py
Calculator agent node.

Uses the LLM to extract a mathematical expression from the user query,
then calls the python_calculator MCP tool to evaluate it.

Two-step process:
  1. LLM extracts a clean math expression from natural language.
  2. MCP calculator evaluates it safely (no arbitrary code execution).

Failure isolation:
  - If expression extraction fails, the raw query is sent to the calculator.
  - If the calculator returns an error, it is surfaced in calculation_output
    so the summariser can report it gracefully.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from src.config import load_config
from src.graph.state import AgentState, AgentStep
from src.llm import get_llm
from src.mcp_client import MCPClient

log = logging.getLogger(__name__)
cfg = load_config()

_SYSTEM_PROMPT = cfg["calculator_agent"]["system_prompt"]

_EXTRACT_PROMPT = (
    "Extract ONLY the mathematical expression needed to answer the user's query. "
    "You may be given research findings that contain real numbers — use them. "
    "Return the expression as a single line with no explanation, no units, "
    "no variable names — only valid Python math syntax. "
    "Examples:\n"
    "  'What is 15% of 200?' → 200 * 0.15\n"
    "  'Square root of 144 plus 3 squared' → sqrt(144) + 3**2\n"
    "  'Compound interest on 1000 at 5% for 3 years' → 1000 * (1 + 0.05) ** 3\n"
    "  Query: 'What is 5% of Germany GDP?' + Research: 'Germany GDP is 4.3 trillion' → 4300000000000 * 0.05\n"
    "If no numeric expression can be formed even with the research context, return the word NONE."
)


async def _evaluate(expression: str) -> dict[str, Any]:
    """Call the MCP calculator tool."""
    async with MCPClient() as client:
        return await client.call_tool("python_calculator", {"expression": expression})


def _extract_expression(query: str, research_context: str = "") -> str | None:
    """Use the LLM to extract a math expression, optionally using research context."""
    try:
        llm = get_llm()
        user_content = f"Query: {query}"
        if research_context:
            user_content += f"\n\nResearch findings (use the numbers here):\n{research_context}"
        messages = [
            SystemMessage(content=_EXTRACT_PROMPT),
            HumanMessage(content=user_content),
        ]
        raw = llm.invoke(messages).content.strip()
        if raw.upper() == "NONE" or not raw:
            return None
        raw = re.sub(r"^```[a-z]*\n?", "", raw).strip("`").strip()
        return raw
    except Exception as e:
        log.warning("[Calculator] expression extraction failed: %s", e)
        return None


def calculator_node(state: AgentState) -> dict[str, Any]:
    """
    LangGraph node — extracts a math expression and evaluates it via MCP.
    Returns partial state dict.
    """
    t0    = time.monotonic()
    query = state["query"]
    log.info("[Calculator] starting  query=%r", query)

    success    = True
    expression = None
    calc_out   = {}

    # ── 1. Extract expression (use research output if available) ─────────────
    research_context = state.get("research_output") or ""
    expression = _extract_expression(query, research_context)
    if expression:
        log.info("[Calculator] extracted expression: %r", expression)
    else:
        log.warning("[Calculator] could not extract expression from query")
        expression = query   # fall back to raw query

    # ── 2. Evaluate via MCP ───────────────────────────────────────────────────
    try:
        calc_out = asyncio.run(_evaluate(expression))
        if "error" in calc_out:
            log.warning("[Calculator] tool error: %s", calc_out)
            success = False
        else:
            log.info("[Calculator] result: %s", calc_out.get("result"))
    except Exception as e:
        log.error("[Calculator] MCP call failed: %s", e)
        calc_out = {"error": "mcp_error", "detail": str(e)}
        success  = False

    # ── 3. Format output for summariser ──────────────────────────────────────
    if "error" not in calc_out:
        calculation_output = (
            f"Expression: {calc_out.get('expression', expression)}\n"
            f"Result: {calc_out.get('result')}"
        )
    else:
        calculation_output = (
            f"Calculation failed.\n"
            f"Expression attempted: {expression}\n"
            f"Error: {calc_out.get('error')} — {calc_out.get('detail', '')}"
        )

    latency_ms = (time.monotonic() - t0) * 1000

    step: AgentStep = {
        "agent_name":  "calculator",
        "tool_called": "python_calculator",
        "input":       expression,
        "output":      calculation_output,
        "latency_ms":  round(latency_ms, 2),
        "success":     success,
    }

    errors = list(state.get("errors", []))
    if not success:
        errors.append(f"calculator_node: {calc_out.get('error', 'unknown error')}")

    return {
        "calculation_output": calculation_output,
        "agent_trace":        state.get("agent_trace", []) + [step],
        "errors":             errors,
    }