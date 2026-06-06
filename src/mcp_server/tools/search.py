"""
src/mcp_server/tools/search.py
DuckDuckGo search tool — exposed via MCP.
Contract defined in docs/mcp_spec.md § duckduckgo_search.
"""

from __future__ import annotations

from typing import Any

import time

from duckduckgo_search import DDGS
from duckduckgo_search.exceptions import DuckDuckGoSearchException
from pydantic import BaseModel, Field, field_validator

from src.config import get

_RETRY_DELAYS = [2, 5]   # seconds between attempts (3 total tries)


# ── Input / Output schemas (mirror mcp_spec.md) ──────────────────────────────

class SearchInput(BaseModel):
    query: str = Field(..., description="The search query string.")
    max_results: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum number of results to return.",
    )

    @field_validator("query")
    @classmethod
    def query_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("query must not be empty")
        return v.strip()


class SearchResult(BaseModel):
    title: str
    snippet: str
    url: str


class SearchOutput(BaseModel):
    results: list[SearchResult]
    result_count: int
    query: str


# ── Tool implementation ───────────────────────────────────────────────────────

def duckduckgo_search(query: str, max_results: int | None = None) -> dict[str, Any]:
    """
    Search the web via DuckDuckGo.
    Returns a dict matching SearchOutput schema (or an error dict).
    """
    cfg_default: int = get("mcp_server", "tools", "duckduckgo_search", "default_max_results") or 5
    max_results = max_results if max_results is not None else cfg_default

    # Validate input
    try:
        inp = SearchInput(query=query, max_results=max_results)
    except ValueError as e:
        return {"error": "validation_error", "detail": str(e), "results": []}

    last_err: str = ""
    for attempt, delay in enumerate([0] + _RETRY_DELAYS):
        if delay:
            time.sleep(delay)
        try:
            with DDGS() as ddgs:
                raw = list(ddgs.text(inp.query, max_results=inp.max_results))
            break                        # success — exit retry loop
        except DuckDuckGoSearchException as e:
            last_err = str(e)
            if attempt == len(_RETRY_DELAYS):
                return {"error": "network_error", "detail": last_err, "results": [], "result_count": 0, "query": inp.query}
        except Exception as e:
            return {"error": "network_error", "detail": str(e), "results": [], "result_count": 0, "query": inp.query}
    else:
        return {"error": "network_error", "detail": last_err, "results": [], "result_count": 0, "query": inp.query}

    results = [
        SearchResult(
            title=r.get("title", ""),
            snippet=r.get("body", ""),
            url=r.get("href", ""),
        )
        for r in raw
    ]

    out = SearchOutput(results=results, result_count=len(results), query=inp.query)
    return out.model_dump()