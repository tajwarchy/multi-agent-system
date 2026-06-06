"""
src/mcp_server/tools/wikipedia.py
Wikipedia lookup tool — exposed via MCP.
Contract defined in docs/mcp_spec.md § wikipedia_lookup.
"""

from __future__ import annotations

from typing import Any

import time

import wikipedia as wiki
from pydantic import BaseModel, Field, field_validator

from src.config import get

_RETRY_DELAYS = [2, 5]


# ── Input / Output schemas ────────────────────────────────────────────────────

class WikiInput(BaseModel):
    topic: str = Field(..., description="The topic or article title to look up.")
    sentences: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of summary sentences to return.",
    )

    @field_validator("topic")
    @classmethod
    def topic_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("topic must not be empty")
        return v.strip()


class WikiOutput(BaseModel):
    title: str
    summary: str
    url: str
    sentences_returned: int


# ── Tool implementation ───────────────────────────────────────────────────────

def wikipedia_lookup(topic: str, sentences: int | None = None) -> dict[str, Any]:
    """
    Look up a topic on Wikipedia and return a summary.
    Returns a dict matching WikiOutput schema (or an error dict).
    """
    cfg_default: int = get("mcp_server", "tools", "wikipedia_lookup", "default_sentences") or 5
    sentences = sentences if sentences is not None else cfg_default

    try:
        inp = WikiInput(topic=topic, sentences=sentences)
    except ValueError as e:
        return {"error": "validation_error", "detail": str(e)}

    last_err: str = ""
    for attempt, delay in enumerate([0] + _RETRY_DELAYS):
        if delay:
            time.sleep(delay)
        try:
            summary = wiki.summary(inp.topic, sentences=inp.sentences, auto_suggest=True)
            page    = wiki.page(inp.topic, auto_suggest=True)
            out = WikiOutput(
                title=page.title,
                summary=summary,
                url=page.url,
                sentences_returned=inp.sentences,
            )
            return out.model_dump()
        except wiki.exceptions.DisambiguationError as e:
            return {"error": "disambiguation", "title": inp.topic, "options": e.options[:8]}
        except wiki.exceptions.PageError:
            return {"error": "page_not_found", "title": inp.topic}
        except Exception as e:
            last_err = str(e)
            if attempt == len(_RETRY_DELAYS):
                return {"error": "network_error", "detail": last_err}
    return {"error": "network_error", "detail": last_err}