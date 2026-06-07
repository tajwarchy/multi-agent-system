"""
src/logging/sqlite_logger.py
Structured SQLite logger — records every agent step in a queryable table.

Schema (agent_steps table):
  id           INTEGER  PRIMARY KEY AUTOINCREMENT
  query_id     TEXT     — UUID shared across all steps of one graph run
  query        TEXT     — original user query
  agent_name   TEXT     — supervisor | research | calculator | summarizer
  tool_called  TEXT     — MCP tool name, or NULL
  input        TEXT     — truncated input to agent/tool
  output       TEXT     — truncated output from agent/tool
  latency_ms   REAL     — wall-clock ms for this step
  success      INTEGER  — 1 = success, 0 = failure
  timestamp    TEXT     — ISO-8601 UTC timestamp
  route        TEXT     — routing decision for this run (set after supervisor)
  errors       TEXT     — JSON list of errors at time of logging

Used by:
  - graph.py: logs each completed AgentState via log_agent_steps()
  - FastAPI /trace/{query_id} endpoint: queries by query_id
"""

from __future__ import annotations

import json
import logging
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Generator

from src.config import load_config
from src.graph.state import AgentState, AgentStep

log = logging.getLogger(__name__)
cfg = load_config()["sqlite"]

_DB_PATH    = Path(cfg["db_path"])
_TABLE_NAME = cfg["table_name"]


# ── Schema ────────────────────────────────────────────────────────────────────

_CREATE_TABLE_SQL = f"""
CREATE TABLE IF NOT EXISTS {_TABLE_NAME} (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    query_id    TEXT    NOT NULL,
    query       TEXT    NOT NULL,
    agent_name  TEXT    NOT NULL,
    tool_called TEXT,
    input       TEXT,
    output      TEXT,
    latency_ms  REAL,
    success     INTEGER NOT NULL DEFAULT 1,
    timestamp   TEXT    NOT NULL,
    route       TEXT,
    errors      TEXT
);
"""

_CREATE_INDEX_SQL = f"""
CREATE INDEX IF NOT EXISTS idx_query_id ON {_TABLE_NAME}(query_id);
"""


# ── DB connection ─────────────────────────────────────────────────────────────

def _ensure_db() -> None:
    """Create the DB file, directory, table, and index if they don't exist."""
    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(_DB_PATH) as conn:
        conn.execute(_CREATE_TABLE_SQL)
        conn.execute(_CREATE_INDEX_SQL)
        conn.commit()
    log.info("[SQLite] DB ready at %s", _DB_PATH.resolve())


@contextmanager
def _get_conn() -> Generator[sqlite3.Connection, None, None]:
    """Context manager: open connection, yield, commit/close."""
    conn = sqlite3.connect(_DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ── Write ─────────────────────────────────────────────────────────────────────

def log_agent_steps(state: AgentState) -> None:
    """
    Write every step in state["agent_trace"] to SQLite.
    Called once per completed graph run (from graph.py after invoke()).
    Non-fatal — errors are logged but never raised.
    """
    try:
        _ensure_db()
        query_id  = state.get("query_id", "unknown")
        query     = state.get("query", "")
        route     = state.get("route", "")
        errors    = json.dumps(state.get("errors", []))
        timestamp = datetime.now(timezone.utc).isoformat()

        rows = []
        for step in state.get("agent_trace", []):
            rows.append((
                query_id,
                query,
                step.get("agent_name", ""),
                step.get("tool_called"),
                _trunc(step.get("input")),
                _trunc(step.get("output")),
                step.get("latency_ms"),
                1 if step.get("success") else 0,
                timestamp,
                route,
                errors,
            ))

        if not rows:
            return

        sql = f"""
            INSERT INTO {_TABLE_NAME}
              (query_id, query, agent_name, tool_called, input, output,
               latency_ms, success, timestamp, route, errors)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        with _get_conn() as conn:
            conn.executemany(sql, rows)

        log.info("[SQLite] logged %d steps for query_id=%s", len(rows), query_id)

    except Exception as e:
        log.error("[SQLite] log_agent_steps failed (non-fatal): %s", e)


# ── Read ──────────────────────────────────────────────────────────────────────

def get_trace(query_id: str) -> list[dict]:
    """
    Fetch all agent steps for a given query_id, ordered by id (insertion order).
    Returns a list of dicts — used by the FastAPI /trace/{query_id} endpoint.
    Returns [] if query_id not found or DB doesn't exist yet.
    """
    try:
        _ensure_db()
        sql = f"""
            SELECT id, query_id, query, agent_name, tool_called, input, output,
                   latency_ms, success, timestamp, route, errors
            FROM {_TABLE_NAME}
            WHERE query_id = ?
            ORDER BY id ASC
        """
        with _get_conn() as conn:
            rows = conn.execute(sql, (query_id,)).fetchall()

        result = []
        for row in rows:
            d = dict(row)
            d["success"] = bool(d["success"])
            try:
                d["errors"] = json.loads(d["errors"] or "[]")
            except Exception:
                d["errors"] = []
            result.append(d)

        return result

    except Exception as e:
        log.error("[SQLite] get_trace failed: %s", e)
        return []


def get_all_query_ids() -> list[str]:
    """
    Return a list of distinct query_ids in the DB, most recent first.
    Used by the /trace endpoint for discovery.
    """
    try:
        _ensure_db()
        sql = f"""
            SELECT DISTINCT query_id, MAX(timestamp) as ts
            FROM {_TABLE_NAME}
            GROUP BY query_id
            ORDER BY ts DESC
        """
        with _get_conn() as conn:
            rows = conn.execute(sql).fetchall()
        return [r["query_id"] for r in rows]
    except Exception as e:
        log.error("[SQLite] get_all_query_ids failed: %s", e)
        return []


# ── Helpers ───────────────────────────────────────────────────────────────────

def _trunc(value: object, max_len: int = 1000) -> str | None:
    """Truncate any value to a string safe for SQLite storage."""
    if value is None:
        return None
    s = str(value)
    return s[:max_len] if len(s) > max_len else s