"""
src/api/routes/trace.py
GET /trace/{query_id}  — returns the full step-by-step agent trace from SQLite.
GET /traces            — lists all recorded query IDs (most recent first).
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

from src.api.models import TraceResponse, TraceStepResponse
from src.logging.sqlite_logger import get_all_query_ids, get_trace

log    = logging.getLogger(__name__)
router = APIRouter()


@router.get(
    "/trace/{query_id}",
    response_model=TraceResponse,
    summary="Get full agent trace for a query",
    description=(
        "Returns every agent step logged to SQLite for the given query_id. "
        "Use the query_id returned by POST /run."
    ),
)
def get_query_trace(query_id: str) -> TraceResponse:
    log.info("[API] GET /trace/%s", query_id)
    rows = get_trace(query_id)

    if not rows:
        raise HTTPException(
            status_code=404,
            detail=f"No trace found for query_id='{query_id}'. "
                   "Run a query first via POST /run.",
        )

    steps = [
        TraceStepResponse(
            id          = row["id"],
            query_id    = row["query_id"],
            query       = row["query"],
            agent_name  = row["agent_name"],
            tool_called = row.get("tool_called"),
            input       = row.get("input"),
            output      = row.get("output"),
            latency_ms  = row.get("latency_ms"),
            success     = row["success"],
            timestamp   = row["timestamp"],
            route       = row.get("route"),
            errors      = row.get("errors", []),
        )
        for row in rows
    ]

    return TraceResponse(
        query_id    = query_id,
        total_steps = len(steps),
        steps       = steps,
    )


@router.get(
    "/traces",
    summary="List all recorded query IDs",
    description="Returns the most recent query IDs stored in SQLite, newest first.",
)
def list_traces() -> dict:
    ids = get_all_query_ids()
    return {"total": len(ids), "query_ids": ids}