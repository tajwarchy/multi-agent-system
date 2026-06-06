"""
src/mcp_client.py
Thin MCP client wrapper.

Agents call tools through this module exclusively — they never import
from src/mcp_server/ directly. This enforces the MCP decoupling contract.

Design:
  - MCPClient spins up the MCP server as a subprocess (stdio transport).
  - Exposes two async methods: list_tools() and call_tool().
  - Used as an async context manager so the subprocess lifecycle is managed.

Usage (inside an agent node):
    async with MCPClient() as client:
        tools = await client.list_tools()
        result = await client.call_tool("duckduckgo_search", {"query": "LangGraph"})
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import Tool

log = logging.getLogger(__name__)

# Path to the server entrypoint
_SERVER_MODULE = "src.mcp_server.server"


class MCPClient:
    """
    Async context manager that manages a connection to the local MCP server.

    Example:
        async with MCPClient() as client:
            result = await client.call_tool("python_calculator", {"expression": "2**10"})
    """

    def __init__(self) -> None:
        self._session: ClientSession | None  = None
        self._exit_stack = None

    async def __aenter__(self) -> "MCPClient":
        from contextlib import AsyncExitStack
        self._exit_stack = AsyncExitStack()
        await self._exit_stack.__aenter__()

        server_params = StdioServerParameters(
            command=sys.executable,          # use the current conda Python
            args=["-m", _SERVER_MODULE],
            env=None,                        # inherit current environment
        )

        stdio_transport = await self._exit_stack.enter_async_context(
            stdio_client(server_params)
        )
        read, write = stdio_transport

        self._session = await self._exit_stack.enter_async_context(
            ClientSession(read, write)
        )
        await self._session.initialize()
        log.debug("MCPClient: session initialized")
        return self

    async def __aexit__(self, *args: Any) -> None:
        if self._exit_stack:
            await self._exit_stack.__aexit__(*args)
        log.debug("MCPClient: session closed")

    # ── Public API ────────────────────────────────────────────────────────────

    async def list_tools(self) -> list[Tool]:
        """Return the list of tools registered on the MCP server."""
        self._ensure_connected()
        resp = await self._session.list_tools()  # type: ignore[union-attr]
        return resp.tools

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """
        Call a named tool and return the parsed JSON result dict.

        On MCP-level errors (unknown tool, server crash), raises RuntimeError.
        Tool-level errors (e.g. page_not_found) are returned as dicts with
        an "error" key — matching the contract in mcp_spec.md.
        """
        self._ensure_connected()
        log.info("MCPClient.call_tool: %s  args=%s", name, arguments)

        resp = await self._session.call_tool(name, arguments)  # type: ignore[union-attr]

        if not resp.content:
            raise RuntimeError(f"MCP tool '{name}' returned empty content")

        raw_text = resp.content[0].text

        try:
            result = json.loads(raw_text)
        except json.JSONDecodeError:
            # If the server returned plain text (shouldn't happen), wrap it
            result = {"text": raw_text}

        if resp.isError:
            log.warning("MCPClient: tool '%s' returned an error: %s", name, result)

        return result

    def _ensure_connected(self) -> None:
        if self._session is None:
            raise RuntimeError("MCPClient must be used as an async context manager.")


# ── Convenience: one-shot call (for tests / scripts) ─────────────────────────

async def call_tool_once(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Spin up a session, call one tool, and shut down. Good for tests."""
    async with MCPClient() as client:
        return await client.call_tool(name, arguments)