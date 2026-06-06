"""
tests/test_mcp_server.py
Integration tests: MCP client ↔ MCP server (subprocess, stdio transport).

These tests spin up the real MCP server as a subprocess through MCPClient
and exercise the full round-trip: client → server → tool → response.

Requires the conda env to be active and config/config.yaml to exist.

Run:
    pytest tests/test_mcp_server.py -v                       # local (network tools skipped)
    pytest tests/test_mcp_server.py -v -m network            # include network tools
"""

from __future__ import annotations

from pathlib import Path

import pytest
import pytest_asyncio

from src.mcp_client import MCPClient, call_tool_once


# ═══════════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture(scope="module")
def sample_data_file():
    """Create a temp file in ./data/ for file_reader tests."""
    Path("data").mkdir(exist_ok=True)
    p = Path("data/integration_test.txt")
    p.write_text("Integration test content.", encoding="utf-8")
    yield "integration_test.txt"
    p.unlink(missing_ok=True)


# ═══════════════════════════════════════════════════════════════════════════════
# list_tools
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_list_tools_returns_four():
    async with MCPClient() as client:
        tools = await client.list_tools()
    names = {t.name for t in tools}
    assert "duckduckgo_search"  in names
    assert "wikipedia_lookup"   in names
    assert "python_calculator"  in names
    assert "local_file_reader"  in names


@pytest.mark.asyncio
async def test_list_tools_have_schemas():
    async with MCPClient() as client:
        tools = await client.list_tools()
    for t in tools:
        assert t.inputSchema is not None
        assert "properties" in t.inputSchema


# ═══════════════════════════════════════════════════════════════════════════════
# python_calculator (no network needed)
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_calculator_via_mcp_basic():
    r = await call_tool_once("python_calculator", {"expression": "2 + 2"})
    assert r.get("result") == 4.0


@pytest.mark.asyncio
async def test_calculator_via_mcp_sqrt():
    r = await call_tool_once("python_calculator", {"expression": "sqrt(81)"})
    assert r.get("result") == 9.0


@pytest.mark.asyncio
async def test_calculator_via_mcp_division_by_zero():
    r = await call_tool_once("python_calculator", {"expression": "10 / 0"})
    assert r.get("error") == "division_by_zero"


@pytest.mark.asyncio
async def test_calculator_via_mcp_disallowed():
    r = await call_tool_once("python_calculator", {"expression": "__import__('os')"})
    assert "error" in r


# ═══════════════════════════════════════════════════════════════════════════════
# local_file_reader (no network needed)
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_file_reader_via_mcp(sample_data_file):
    r = await call_tool_once("local_file_reader", {"file_path": sample_data_file})
    assert r.get("content") == "Integration test content."


@pytest.mark.asyncio
async def test_file_reader_via_mcp_not_found():
    r = await call_tool_once("local_file_reader", {"file_path": "nonexistent.txt"})
    assert r.get("error") == "file_not_found"


@pytest.mark.asyncio
async def test_file_reader_via_mcp_traversal():
    r = await call_tool_once("local_file_reader", {"file_path": "../config/config.yaml"})
    assert r.get("error") == "access_denied"


@pytest.mark.asyncio
async def test_unknown_tool_returns_error():
    async with MCPClient() as client:
        r = await client.call_tool("nonexistent_tool", {})
    assert "error" in r or "Unknown" in str(r)


# ═══════════════════════════════════════════════════════════════════════════════
# DuckDuckGo + Wikipedia (network)
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
@pytest.mark.network
async def test_search_via_mcp():
    r = await call_tool_once("duckduckgo_search", {"query": "LangGraph", "max_results": 3})
    if r.get("error") == "network_error":
        pytest.skip("DuckDuckGo rate-limited — transient, not a code bug")
    assert r.get("result_count", 0) > 0


@pytest.mark.asyncio
@pytest.mark.network
async def test_wikipedia_via_mcp():
    r = await call_tool_once("wikipedia_lookup", {"topic": "Artificial intelligence", "sentences": 3})
    if r.get("error") == "network_error":
        pytest.skip("Wikipedia rate-limited — transient, not a code bug")
    assert len(r.get("summary", "")) > 0