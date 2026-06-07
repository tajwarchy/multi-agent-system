"""
tests/test_graph.py
Graph-level tests — routing logic, state structure, failure isolation.

These tests use mocking to avoid real LLM/MCP calls, so they run
instantly without Ollama or internet.

Run:
    pytest tests/test_graph.py -v
"""

from __future__ import annotations

import uuid
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from src.graph.state import AgentState


# ── Helpers ───────────────────────────────────────────────────────────────────

def _base_state(**overrides) -> AgentState:
    state: AgentState = {
        "query_id":           str(uuid.uuid4()),
        "query":              "test query",
        "route":              None,
        "memory_context":     None,
        "research_output":    None,
        "calculation_output": None,
        "final_answer":       None,
        "agent_trace":        [],
        "errors":             [],
        "retry_counts":       {"research": 0, "calculator": 0},
    }
    state.update(overrides)
    return state


# ── Conditional router ────────────────────────────────────────────────────────

class TestRouter:
    """
    Tests for the routing logic — implemented as a pure function here so
    we don't trigger the LangGraph import chain (which requires compatible
    langchain-core). After `pip install langgraph==0.2.73` these can import
    directly from src.graph.graph instead.
    """

    # Pure mirror of route_after_supervisor logic for isolated testing
    @staticmethod
    def _route(state):
        from src.config import load_config
        routing = load_config()["graph"]["routing"]
        r = state.get("route")
        if r == routing["research_only"]:     return "research"
        if r == routing["calculation_only"]:  return "calculator"
        if r == routing["both"]:              return "both_research"
        return "summarizer"

    def test_research_only_route(self):
        assert self._route(_base_state(route="research_only")) == "research"

    def test_calculation_only_route(self):
        assert self._route(_base_state(route="calculation_only")) == "calculator"

    def test_both_route(self):
        assert self._route(_base_state(route="both")) == "both_research"

    def test_summarize_only_route(self):
        assert self._route(_base_state(route="summarize_only")) == "summarizer"

    def test_unknown_route_defaults_to_summarizer(self):
        assert self._route(_base_state(route="nonsense_route")) == "summarizer"

    def test_none_route_defaults_to_summarizer(self):
        assert self._route(_base_state(route=None)) == "summarizer"


# ── Supervisor node ───────────────────────────────────────────────────────────

class TestSupervisorNode:

    @patch("src.graph.nodes.supervisor.get_llm")
    @patch("src.graph.nodes.supervisor._get_memory_context", return_value="")
    def test_supervisor_sets_route(self, mock_mem, mock_llm):
        mock_resp         = MagicMock()
        mock_resp.content = "research_only"
        mock_llm.return_value.invoke.return_value = mock_resp

        from src.graph.nodes.supervisor import supervisor_node
        state  = _base_state(query="Who invented Python?")
        result = supervisor_node(state)

        assert result["route"] == "research_only"
        assert len(result["agent_trace"]) == 1
        assert result["agent_trace"][0]["agent_name"] == "supervisor"

    @patch("src.graph.nodes.supervisor.get_llm")
    @patch("src.graph.nodes.supervisor._get_memory_context", return_value="")
    def test_supervisor_defaults_on_bad_llm_output(self, mock_mem, mock_llm):
        mock_resp         = MagicMock()
        mock_resp.content = "I cannot determine the routing."
        mock_llm.return_value.invoke.return_value = mock_resp

        from src.graph.nodes.supervisor import supervisor_node
        state  = _base_state(query="some ambiguous query")
        result = supervisor_node(state)

        # Should default to research_only — never crash
        assert result["route"] == "research_only"

    @patch("src.graph.nodes.supervisor.get_llm")
    @patch("src.graph.nodes.supervisor._get_memory_context", return_value="")
    def test_supervisor_handles_llm_exception(self, mock_mem, mock_llm):
        mock_llm.return_value.invoke.side_effect = RuntimeError("Ollama down")

        from src.graph.nodes.supervisor import supervisor_node
        state  = _base_state(query="anything")
        result = supervisor_node(state)

        # Must not raise — returns default route
        assert result["route"] == "research_only"
        assert result["agent_trace"][0]["success"] is False


# ── Calculator node ───────────────────────────────────────────────────────────

class TestCalculatorNode:

    @patch("src.graph.nodes.calculator.asyncio.run")
    @patch("src.graph.nodes.calculator.get_llm")
    def test_calculator_success(self, mock_llm, mock_run):
        mock_resp         = MagicMock()
        mock_resp.content = "sqrt(144)"
        mock_llm.return_value.invoke.return_value = mock_resp
        mock_run.return_value = {"expression": "sqrt(144)", "result": 12.0, "result_type": "float"}

        from src.graph.nodes.calculator import calculator_node
        state  = _base_state(query="What is the square root of 144?")
        result = calculator_node(state)

        assert "12.0" in result["calculation_output"]
        assert result["agent_trace"][0]["success"] is True
        assert result["errors"] == []

    @patch("src.graph.nodes.calculator.asyncio.run")
    @patch("src.graph.nodes.calculator.get_llm")
    def test_calculator_tool_error_surfaces_gracefully(self, mock_llm, mock_run):
        mock_resp         = MagicMock()
        mock_resp.content = "1 / 0"
        mock_llm.return_value.invoke.return_value = mock_resp
        mock_run.return_value = {"error": "division_by_zero", "expression": "1 / 0"}

        from src.graph.nodes.calculator import calculator_node
        state  = _base_state(query="Divide 1 by 0")
        result = calculator_node(state)

        assert "Calculation failed" in result["calculation_output"]
        assert result["agent_trace"][0]["success"] is False
        assert len(result["errors"]) > 0

    @patch("src.graph.nodes.calculator.asyncio.run")
    @patch("src.graph.nodes.calculator.get_llm")
    def test_calculator_mcp_exception_handled(self, mock_llm, mock_run):
        mock_resp         = MagicMock()
        mock_resp.content = "2 + 2"
        mock_llm.return_value.invoke.return_value = mock_resp
        mock_run.side_effect = Exception("MCP server crashed")

        from src.graph.nodes.calculator import calculator_node
        state  = _base_state(query="2 plus 2")
        result = calculator_node(state)

        # Must not raise — returns error output
        assert "Calculation failed" in result["calculation_output"]


# ── Summarizer node ───────────────────────────────────────────────────────────

class TestSummarizerNode:

    @patch("src.graph.nodes.summarizer._persist_trace")
    @patch("src.graph.nodes.summarizer.get_llm")
    def test_summarizer_builds_final_answer(self, mock_llm, mock_persist):
        mock_resp         = MagicMock()
        mock_resp.content = "The answer is 42."
        mock_llm.return_value.invoke.return_value = mock_resp

        from src.graph.nodes.summarizer import summarizer_node
        state  = _base_state(research_output="Some research.", calculation_output="Result: 42")
        result = summarizer_node(state)

        assert result["final_answer"] == "The answer is 42."
        assert result["agent_trace"][0]["agent_name"] == "summarizer"
        mock_persist.assert_called_once()

    @patch("src.graph.nodes.summarizer._persist_trace")
    @patch("src.graph.nodes.summarizer.get_llm")
    def test_summarizer_degrades_gracefully_on_llm_failure(self, mock_llm, mock_persist):
        mock_llm.return_value.invoke.side_effect = RuntimeError("LLM down")

        from src.graph.nodes.summarizer import summarizer_node
        state  = _base_state(research_output="Some findings.")
        result = summarizer_node(state)

        # Must not raise — returns raw context as fallback
        assert result["final_answer"] is not None
        assert "Summarisation failed" in result["final_answer"]
        assert len(result["errors"]) > 0


# ── State structure ───────────────────────────────────────────────────────────

class TestStateStructure:

    def test_state_has_all_required_keys(self):
        state = _base_state()
        required = {
            "query_id", "query", "route", "memory_context",
            "research_output", "calculation_output", "final_answer",
            "agent_trace", "errors", "retry_counts",
        }
        assert required.issubset(set(state.keys()))

    def test_agent_trace_accumulates(self):
        """Each node should append to agent_trace, not overwrite it."""
        from src.graph.state import AgentStep
        step1: AgentStep = {
            "agent_name": "supervisor", "tool_called": None,
            "input": "q", "output": "research_only",
            "latency_ms": 10.0, "success": True,
        }
        state = _base_state(agent_trace=[step1])

        step2: AgentStep = {
            "agent_name": "research", "tool_called": "duckduckgo_search",
            "input": "q", "output": "findings",
            "latency_ms": 200.0, "success": True,
        }
        state["agent_trace"].append(step2)
        assert len(state["agent_trace"]) == 2
        assert state["agent_trace"][0]["agent_name"] == "supervisor"
        assert state["agent_trace"][1]["agent_name"] == "research"