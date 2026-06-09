"""
src/api/routes/run.py
POST /run — runs the full multi-agent graph and returns the result.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

from src.api.models import AgentStepResponse, RunRequest, RunResponse
from src.graph.graph import run_graph

log    = logging.getLogger(__name__)
router = APIRouter()


@router.post(
    "/run",
    response_model=RunResponse,
    summary="Run the multi-agent graph",
    description=(
        "Submit a natural-language query. The supervisor agent classifies it, "
        "routes it to specialist agents (research, calculator, or both), and "
        "returns the synthesised final answer along with the full agent trace."
    ),
)
def run_query(req: RunRequest) -> RunResponse:
    log.info("[API] POST /run  query=%r  query_id=%s", req.query, req.query_id)

    try:
        result = run_graph(query=req.query, query_id=req.query_id)
    except Exception as e:
        log.exception("[API] run_graph raised an unhandled exception")
        raise HTTPException(status_code=500, detail=f"Graph execution failed: {e}")

    trace = [
        AgentStepResponse(
            agent_name  = step.get("agent_name", ""),
            tool_called = step.get("tool_called"),
            input       = str(step.get("input",  ""))[:500],
            output      = str(step.get("output", ""))[:500],
            latency_ms  = step.get("latency_ms", 0.0),
            success     = step.get("success", True),
        )
        for step in result.get("agent_trace", [])
    ]

    return RunResponse(
        query_id     = result["query_id"],
        query        = result["query"],
        route        = result.get("route"),
        final_answer = result.get("final_answer"),
        agent_trace  = trace,
        errors       = result.get("errors", []),
        total_steps  = len(trace),
    )