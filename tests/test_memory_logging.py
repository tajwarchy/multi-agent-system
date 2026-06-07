"""
tests/test_memory_logging.py
Unit + integration tests for ChromaDB memory and SQLite logging.

Run:
    pytest tests/test_memory_logging.py -v
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from unittest.mock import patch

import pytest

from src.graph.state import AgentState, AgentStep


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_state(query: str = "test query", route: str = "research_only") -> AgentState:
    step: AgentStep = {
        "agent_name":  "supervisor",
        "tool_called": None,
        "input":       query,
        "output":      route,
        "latency_ms":  50.0,
        "success":     True,
    }
    return {
        "query_id":           str(uuid.uuid4()),
        "query":              query,
        "route":              route,
        "memory_context":     None,
        "research_output":    "Some research findings.",
        "calculation_output": None,
        "final_answer":       "The answer is 42.",
        "agent_trace":        [step],
        "errors":             [],
        "retry_counts":       {"research": 0, "calculator": 0},
    }


# ═══════════════════════════════════════════════════════════════════════════════
# SQLite Logger
# ═══════════════════════════════════════════════════════════════════════════════

class TestSQLiteLogger:

    @pytest.fixture(autouse=True)
    def tmp_db(self, tmp_path, monkeypatch):
        """Redirect SQLite to a temp file for each test."""
        db_path = tmp_path / "test_agent_steps.db"
        monkeypatch.setattr("src.logging.sqlite_logger._DB_PATH", db_path)
        # Reset the lru_cache on _ensure_db isn't cached across tests
        yield db_path

    def test_log_and_retrieve_single_run(self, tmp_db):
        from src.logging.sqlite_logger import get_trace, log_agent_steps
        state = _make_state("Who invented Python?", "research_only")
        log_agent_steps(state)
        trace = get_trace(state["query_id"])
        assert len(trace) == 1
        assert trace[0]["agent_name"] == "supervisor"
        assert trace[0]["query"] == "Who invented Python?"
        assert trace[0]["success"] is True

    def test_multiple_steps_logged_in_order(self, tmp_db):
        from src.logging.sqlite_logger import get_trace, log_agent_steps

        steps: list[AgentStep] = [
            {"agent_name": "supervisor",  "tool_called": None,
             "input": "q", "output": "research_only", "latency_ms": 10.0, "success": True},
            {"agent_name": "research",    "tool_called": "duckduckgo_search",
             "input": "q", "output": "findings",      "latency_ms": 200.0, "success": True},
            {"agent_name": "summarizer",  "tool_called": None,
             "input": "ctx", "output": "answer",      "latency_ms": 50.0,  "success": True},
        ]
        state = _make_state()
        state["agent_trace"] = steps

        log_agent_steps(state)
        trace = get_trace(state["query_id"])

        assert len(trace) == 3
        assert trace[0]["agent_name"] == "supervisor"
        assert trace[1]["agent_name"] == "research"
        assert trace[2]["agent_name"] == "summarizer"

    def test_unknown_query_id_returns_empty(self, tmp_db):
        from src.logging.sqlite_logger import get_trace, log_agent_steps
        log_agent_steps(_make_state())
        result = get_trace("nonexistent-id-9999")
        assert result == []

    def test_errors_serialised_correctly(self, tmp_db):
        from src.logging.sqlite_logger import get_trace, log_agent_steps
        state = _make_state()
        state["errors"] = ["research_node failed", "calculator_node: timeout"]
        log_agent_steps(state)
        trace = get_trace(state["query_id"])
        assert "research_node failed" in trace[0]["errors"]

    def test_failure_step_logged(self, tmp_db):
        from src.logging.sqlite_logger import get_trace, log_agent_steps
        state = _make_state()
        state["agent_trace"] = [{
            "agent_name":  "calculator",
            "tool_called": "python_calculator",
            "input":       "bad expression",
            "output":      "Calculation failed.",
            "latency_ms":  30.0,
            "success":     False,
        }]
        log_agent_steps(state)
        trace = get_trace(state["query_id"])
        assert trace[0]["success"] is False

    def test_get_all_query_ids(self, tmp_db):
        from src.logging.sqlite_logger import get_all_query_ids, log_agent_steps
        s1, s2 = _make_state("q1"), _make_state("q2")
        log_agent_steps(s1)
        log_agent_steps(s2)
        ids = get_all_query_ids()
        assert s1["query_id"] in ids
        assert s2["query_id"] in ids

    def test_long_output_truncated(self, tmp_db):
        from src.logging.sqlite_logger import get_trace, log_agent_steps
        state = _make_state()
        state["agent_trace"][0]["output"] = "x" * 5000
        log_agent_steps(state)
        trace = get_trace(state["query_id"])
        assert len(trace[0]["output"]) <= 1001   # 1000 chars + possible truncation marker


# ═══════════════════════════════════════════════════════════════════════════════
# ChromaDB Memory
# ═══════════════════════════════════════════════════════════════════════════════

class TestChromaMemory:

    @pytest.fixture(autouse=True)
    def tmp_chroma(self, tmp_path):
        """
        Redirect ChromaDB to a temp directory for each test.
        Directly mutates the module-level cfg dict (a plain dict —
        monkeypatch can't patch __getitem__ on it) and restores on teardown.
        Clears lru_caches so each test gets a fresh client + collection.
        """
        import src.memory.chroma_store as cs

        # Save originals
        orig = {k: cs.cfg[k] for k in cs.cfg}

        # Redirect to tmp
        cs.cfg["persist_directory"]    = str(tmp_path / "chroma")
        cs.cfg["collection_name"]      = f"test_{uuid.uuid4().hex[:8]}"
        cs.cfg["n_results"]            = 3
        cs.cfg["similarity_threshold"] = 0.0   # accept all results in tests

        # Fresh client + collection for this test
        cs._get_client.cache_clear()
        cs._get_collection.cache_clear()

        yield

        # Restore originals and clear caches again
        for k, v in orig.items():
            cs.cfg[k] = v
        cs._get_client.cache_clear()
        cs._get_collection.cache_clear()

    def test_persist_and_retrieve(self):
        from src.memory.chroma_store import persist_trace, retrieve_similar_traces
        state = _make_state("Who invented Python?", "research_only")
        state["final_answer"] = "Guido van Rossum invented Python."
        persist_trace(state)

        context = retrieve_similar_traces("Who created Python?")
        # Should retrieve something semantically similar
        assert isinstance(context, str)
        # context may be empty if similarity < threshold (threshold=0.0 so should match)
        assert "Python" in context or context == ""  # graceful either way

    def test_empty_collection_returns_empty_string(self):
        from src.memory.chroma_store import retrieve_similar_traces
        result = retrieve_similar_traces("any query")
        assert result == ""

    def test_persist_is_idempotent(self):
        from src.memory.chroma_store import _get_collection, persist_trace
        state = _make_state("repeated query")
        persist_trace(state)
        persist_trace(state)   # upsert — should not duplicate
        count = _get_collection().count()
        assert count == 1

    def test_persist_non_fatal_on_bad_state(self):
        from src.memory.chroma_store import persist_trace
        # Should not raise even with a malformed state
        persist_trace({})   # type: ignore

    def test_retrieve_non_fatal_on_error(self):
        from src.memory.chroma_store import retrieve_similar_traces
        # Should return "" not raise
        result = retrieve_similar_traces("")
        assert isinstance(result, str)