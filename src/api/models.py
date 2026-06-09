"""
src/api/models.py
Pydantic request/response models for the FastAPI layer.
These are the public API contracts — separate from internal AgentState.
"""

from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field


# ── POST /run ─────────────────────────────────────────────────────────────────

class RunRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="The natural-language query to run through the agent graph.",
        examples=["What is the square root of 256?", "Who invented the internet?"],
    )
    query_id: Optional[str] = Field(
        default=None,
        description="Optional UUID. Auto-generated if not provided.",
    )


class AgentStepResponse(BaseModel):
    agent_name:  str
    tool_called: Optional[str]
    input:       Optional[str]
    output:      Optional[str]
    latency_ms:  float
    success:     bool


class RunResponse(BaseModel):
    query_id:           str
    query:              str
    route:              Optional[str]
    final_answer:       Optional[str]
    agent_trace:        list[AgentStepResponse]
    errors:             list[str]
    total_steps:        int


# ── GET /trace/{query_id} ─────────────────────────────────────────────────────

class TraceStepResponse(BaseModel):
    id:          int
    query_id:    str
    query:       str
    agent_name:  str
    tool_called: Optional[str]
    input:       Optional[str]
    output:      Optional[str]
    latency_ms:  Optional[float]
    success:     bool
    timestamp:   str
    route:       Optional[str]
    errors:      list[str]


class TraceResponse(BaseModel):
    query_id:    str
    total_steps: int
    steps:       list[TraceStepResponse]


# ── GET /health ───────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status:        str
    chroma_docs:   int
    sqlite_runs:   int