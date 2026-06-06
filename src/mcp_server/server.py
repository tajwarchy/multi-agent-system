"""
src/mcp_server/server.py
Local MCP server — exposes 4 tools to any MCP-compatible client.

Start with:
    python -m src.mcp_server.server

Handler signatures match MCP Python SDK 1.9.x:
  - @app.list_tools()  → async def handler() -> list[Tool]
  - @app.call_tool()   → async def handler(name: str, arguments: dict) -> list[TextContent]
"""

from __future__ import annotations

import json
import logging

import anyio
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

from src.config import get, load_config
from src.mcp_server.tools.calculator  import python_calculator
from src.mcp_server.tools.file_reader import local_file_reader
from src.mcp_server.tools.search      import duckduckgo_search
from src.mcp_server.tools.wikipedia   import wikipedia_lookup

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s | MCP | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# ── Server instance ───────────────────────────────────────────────────────────
cfg   = load_config()
s_cfg = cfg["mcp_server"]
app   = Server(s_cfg["server_name"])

# ── Tool registry ─────────────────────────────────────────────────────────────
_TOOLS: list[tuple[Tool, callable]] = []

if get("mcp_server", "tools", "duckduckgo_search", "enabled"):
    _TOOLS.append((
        Tool(
            name="duckduckgo_search",
            description=(
                "Search the web using DuckDuckGo. Returns top N results as "
                "title + snippet + URL objects. Use for real-time information retrieval."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query":       {"type": "string",  "description": "The search query string."},
                    "max_results": {"type": "integer", "description": "Max results (1-20). Default 5.", "default": 5},
                },
                "required": ["query"],
            },
        ),
        duckduckgo_search,
    ))

if get("mcp_server", "tools", "wikipedia_lookup", "enabled"):
    _TOOLS.append((
        Tool(
            name="wikipedia_lookup",
            description=(
                "Look up a topic on Wikipedia and return a plain-text summary. "
                "Use for factual, encyclopedic information."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "topic":     {"type": "string",  "description": "Topic or article title."},
                    "sentences": {"type": "integer", "description": "Summary sentences (1-20). Default 5.", "default": 5},
                },
                "required": ["topic"],
            },
        ),
        wikipedia_lookup,
    ))

if get("mcp_server", "tools", "python_calculator", "enabled"):
    _TOOLS.append((
        Tool(
            name="python_calculator",
            description=(
                "Evaluate a mathematical expression safely. Supports: +, -, *, /, **, %, "
                "sqrt(), abs(), round(), floor(), ceil(), pow(), log(), log10(), pi, e. "
                "No imports, no variables, no function definitions."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Math expression to evaluate. Example: 'sqrt(144) + round(3.14, 1)'",
                    },
                },
                "required": ["expression"],
            },
        ),
        python_calculator,
    ))

if get("mcp_server", "tools", "local_file_reader", "enabled"):
    _TOOLS.append((
        Tool(
            name="local_file_reader",
            description=(
                "Read a local file from the sandboxed ./data/ directory. "
                "Provide a relative path such as 'report.txt' or 'subdir/notes.md'. "
                "Absolute paths and path traversal are rejected."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Relative path inside ./data/."},
                    "encoding":  {"type": "string", "description": "File encoding. Default 'utf-8'.", "default": "utf-8"},
                },
                "required": ["file_path"],
            },
        ),
        local_file_reader,
    ))

_TOOL_MAP: dict[str, callable] = {t.name: fn for t, fn in _TOOLS}


# ── MCP handlers (SDK 1.9.x signatures) ──────────────────────────────────────

@app.list_tools()
async def handle_list_tools() -> list[Tool]:
    """No argument — SDK 1.9.x calls this with zero args."""
    log.info("list_tools called — returning %d tools", len(_TOOLS))
    return [t for t, _ in _TOOLS]


@app.call_tool()
async def handle_call_tool(name: str, arguments: dict) -> list[TextContent]:
    """SDK 1.9.x passes (name, arguments) as positional args, not a request object."""
    arguments = arguments or {}
    log.info("call_tool: %s  args=%s", name, arguments)

    fn = _TOOL_MAP.get(name)
    if fn is None:
        err = f"Unknown tool: '{name}'. Available: {list(_TOOL_MAP)}"
        log.error(err)
        return [TextContent(type="text", text=json.dumps({"error": "unknown_tool", "detail": err}))]

    try:
        result = fn(**arguments)
        return [TextContent(type="text", text=json.dumps(result, ensure_ascii=False))]
    except TypeError as e:
        err = f"Bad arguments for tool '{name}': {e}"
        log.error(err)
        return [TextContent(type="text", text=json.dumps({"error": "bad_arguments", "detail": err}))]
    except Exception as e:
        err = f"Tool '{name}' raised an unexpected error: {e}"
        log.exception(err)
        return [TextContent(type="text", text=json.dumps({"error": "internal_error", "detail": err}))]


# ── Entrypoint ────────────────────────────────────────────────────────────────

async def _main() -> None:
    log.info(
        "Starting MCP server '%s' v%s  (%d tools)",
        s_cfg["server_name"],
        s_cfg["server_version"],
        len(_TOOLS),
    )
    for t, _ in _TOOLS:
        log.info("  tool: %s", t.name)

    async with stdio_server() as (r, w):
        await app.run(r, w, app.create_initialization_options())


if __name__ == "__main__":
    anyio.run(_main)