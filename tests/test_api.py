"""
tests/test_api.py
FastAPI endpoint tests — all graph calls are mocked so these run
instantly with no LLM or MCP server needed.

Run:
    pytest tests/test_api.py -v
"""

from __future__ import annotations

import uuid
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app)

# ── Shared mock result ────────────────────────────────────────────────────────

def _mock_result(query: str = "test", query_id: str | None = None) -> dict:
    qid = query_id or str(uuid.uuid4())
    return {
        "query_id":           qid,
        "query":              query,
        "route":              "calculation_only",
        "memory_context":     None,
        "research_output":    None,
        "calculation_output": "Expression: sqrt(4)\nResult: 2.0",
        "final_answer":       "The answer is 2.0.",
        "agent_trace": [
            {"agent_name": "supervisor",  "tool_called": None,
             "input": query, "output": "calculation_only",
             "latency_ms": 50.0,  "success": True},
            {"agent_name": "calculator",  "tool_called": "python_calculator",
             "input": "sqrt(4)", "output": "Result: 2.0",
             "latency_ms": 30.0,  "success": True},
            {"agent_name": "summarizer",  "tool_called": None,
             "input": "ctx", "output": "The answer is 2.0.",
             "latency_ms": 20.0,  "success": True},
        ],
        "errors":       [],
        "retry_counts": {"research": 0, "calculator": 0},
    }


# ═══════════════════════════════════════════════════════════════════════════════
# GET /health
# ═══════════════════════════════════════════════════════════════════════════════

def test_health_returns_ok():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
    assert "chroma_docs" in r.json()
    assert "sqlite_runs" in r.json()


def test_root_returns_endpoints():
    r = client.get("/")
    assert r.status_code == 200
    assert "/run" in r.json()["endpoints"]


# ═══════════════════════════════════════════════════════════════════════════════
# POST /run
# ═══════════════════════════════════════════════════════════════════════════════

class TestRunEndpoint:

    @patch("src.api.routes.run.run_graph")
    def test_run_returns_200_with_answer(self, mock_graph):
        mock_graph.return_value = _mock_result("What is sqrt(4)?")
        r = client.post("/run", json={"query": "What is sqrt(4)?"})
        assert r.status_code == 200
        body = r.json()
        assert body["final_answer"] == "The answer is 2.0."
        assert body["route"] == "calculation_only"
        assert body["total_steps"] == 3
        assert len(body["agent_trace"]) == 3
        assert body["errors"] == []

    @patch("src.api.routes.run.run_graph")
    def test_run_returns_query_id(self, mock_graph):
        fixed_id = str(uuid.uuid4())
        mock_graph.return_value = _mock_result(query_id=fixed_id)
        r = client.post("/run", json={"query": "test", "query_id": fixed_id})
        assert r.json()["query_id"] == fixed_id

    @patch("src.api.routes.run.run_graph")
    def test_run_trace_has_correct_fields(self, mock_graph):
        mock_graph.return_value = _mock_result()
        r = client.post("/run", json={"query": "test"})
        step = r.json()["agent_trace"][0]
        assert "agent_name"  in step
        assert "tool_called" in step
        assert "latency_ms"  in step
        assert "success"     in step

    def test_run_empty_query_returns_422(self):
        r = client.post("/run", json={"query": ""})
        assert r.status_code == 422

    def test_run_missing_query_returns_422(self):
        r = client.post("/run", json={})
        assert r.status_code == 422

    def test_run_query_too_long_returns_422(self):
        r = client.post("/run", json={"query": "x" * 2001})
        assert r.status_code == 422

    @patch("src.api.routes.run.run_graph")
    def test_run_graph_exception_returns_500(self, mock_graph):
        mock_graph.side_effect = RuntimeError("Ollama crashed")
        r = client.post("/run", json={"query": "test"})
        assert r.status_code == 500
        assert "Graph execution failed" in r.json()["detail"]

    @patch("src.api.routes.run.run_graph")
    def test_run_with_errors_still_returns_200(self, mock_graph):
        result = _mock_result()
        result["errors"] = ["calculator_node: syntax_error"]
        mock_graph.return_value = result
        r = client.post("/run", json={"query": "test"})
        assert r.status_code == 200
        assert len(r.json()["errors"]) == 1


# ═══════════════════════════════════════════════════════════════════════════════
# GET /trace/{query_id}
# ═══════════════════════════════════════════════════════════════════════════════

class TestTraceEndpoint:

    @patch("src.api.routes.trace.get_trace")
    def test_trace_returns_steps(self, mock_get_trace):
        qid = str(uuid.uuid4())
        mock_get_trace.return_value = [
            {"id": 1, "query_id": qid, "query": "test",
             "agent_name": "supervisor", "tool_called": None,
             "input": "test", "output": "calculation_only",
             "latency_ms": 50.0, "success": True,
             "timestamp": "2024-01-01T00:00:00+00:00",
             "route": "calculation_only", "errors": []},
        ]
        r = client.get(f"/trace/{qid}")
        assert r.status_code == 200
        body = r.json()
        assert body["query_id"] == qid
        assert body["total_steps"] == 1
        assert body["steps"][0]["agent_name"] == "supervisor"

    @patch("src.api.routes.trace.get_trace")
    def test_trace_not_found_returns_404(self, mock_get_trace):
        mock_get_trace.return_value = []
        r = client.get("/trace/nonexistent-id")
        assert r.status_code == 404
        assert "No trace found" in r.json()["detail"]

    @patch("src.api.routes.trace.get_all_query_ids")
    def test_list_traces_returns_ids(self, mock_list):
        ids = [str(uuid.uuid4()) for _ in range(3)]
        mock_list.return_value = ids
        r = client.get("/traces")
        assert r.status_code == 200
        assert r.json()["total"] == 3
        assert r.json()["query_ids"] == ids

    @patch("src.api.routes.trace.get_all_query_ids")
    def test_list_traces_empty(self, mock_list):
        mock_list.return_value = []
        r = client.get("/traces")
        assert r.status_code == 200
        assert r.json()["total"] == 0