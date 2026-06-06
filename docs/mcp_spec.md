# MCP Tool Contract — `mcp_spec.md`

This document defines the **Model Context Protocol (MCP) contract** for the local MCP server
used in this project. Every tool exposed by the server is documented here **before any code is
written**. Agents interact with tools exclusively through this interface — they never import tool
implementations directly.

---

## What is MCP?

MCP (Model Context Protocol) is an open protocol that decouples **tool implementation** from
**agent logic**. The MCP server owns and exposes tools. Agents discover and call tools through a
standardised client interface, without knowing anything about how the tools are implemented.

This means:
- Any agent can call any tool without importing it.
- Tools can be swapped, updated, or replaced without touching agent code.
- The MCP server is a self-contained, independently testable service.

---

## Server Configuration

| Property        | Value                     |
|-----------------|---------------------------|
| Transport       | `stdio` (local process)   |
| Server name     | `multi-agent-tool-server` |
| Server version  | `1.0.0`                   |
| Protocol        | MCP 1.0                   |

---

## Tools Exposed

### 1. `duckduckgo_search`

**Description:** Searches the web using DuckDuckGo and returns the top N results as a list of
title + snippet + URL objects. Used by the Research Agent for real-time information retrieval.

**Input Schema:**

```json
{
  "query": {
    "type": "string",
    "description": "The search query string.",
    "required": true
  },
  "max_results": {
    "type": "integer",
    "description": "Maximum number of results to return. Defaults to 5.",
    "required": false,
    "default": 5,
    "minimum": 1,
    "maximum": 20
  }
}
```

**Output Schema:**

```json
{
  "results": [
    {
      "title": "string",
      "snippet": "string",
      "url": "string"
    }
  ],
  "result_count": "integer",
  "query": "string"
}
```

**Error Behavior:**

| Condition                     | Behavior                                              |
|-------------------------------|-------------------------------------------------------|
| Network unavailable           | Returns `{"error": "network_error", "results": []}`   |
| No results found              | Returns `{"results": [], "result_count": 0}`          |
| `query` is empty string       | Raises `ValidationError` before calling DuckDuckGo   |

---

### 2. `wikipedia_lookup`

**Description:** Looks up a topic on Wikipedia and returns a summary (first N sentences of the
article). Used by the Research Agent for factual, encyclopedic information.

**Input Schema:**

```json
{
  "topic": {
    "type": "string",
    "description": "The topic or article title to look up on Wikipedia.",
    "required": true
  },
  "sentences": {
    "type": "integer",
    "description": "Number of summary sentences to return. Defaults to 5.",
    "required": false,
    "default": 5,
    "minimum": 1,
    "maximum": 20
  }
}
```

**Output Schema:**

```json
{
  "title": "string",
  "summary": "string",
  "url": "string",
  "sentences_returned": "integer"
}
```

**Error Behavior:**

| Condition                        | Behavior                                                          |
|----------------------------------|-------------------------------------------------------------------|
| Page not found                   | Returns `{"error": "page_not_found", "title": "<topic>"}`         |
| Disambiguation page hit          | Returns `{"error": "disambiguation", "options": ["...", "..."]}`  |
| `topic` is empty string          | Raises `ValidationError` before querying Wikipedia               |
| Network unavailable              | Returns `{"error": "network_error"}`                              |

---

### 3. `python_calculator`

**Description:** Evaluates a mathematical expression and returns the numeric result. Expressions
are parsed by a sandboxed evaluator — no arbitrary code execution is permitted. Used by the
Calculator Agent for numerical reasoning tasks.

**Input Schema:**

```json
{
  "expression": {
    "type": "string",
    "description": "A mathematical expression to evaluate. Supports: +, -, *, /, **, %, sqrt(), abs(), round(), floor(), ceil(). No imports, no function definitions, no variable assignment.",
    "required": true,
    "examples": ["2 + 2", "sqrt(144)", "round(3.14159, 2)", "(100 * 1.08) ** 2"]
  }
}
```

**Output Schema:**

```json
{
  "expression": "string",
  "result": "number",
  "result_type": "string"
}
```

**Error Behavior:**

| Condition                          | Behavior                                                              |
|------------------------------------|-----------------------------------------------------------------------|
| Division by zero                   | Returns `{"error": "division_by_zero", "expression": "<expr>"}`      |
| Disallowed operation (e.g. import) | Returns `{"error": "disallowed_operation", "expression": "<expr>"}`  |
| Malformed expression               | Returns `{"error": "syntax_error", "expression": "<expr>"}`          |
| Overflow / undefined result        | Returns `{"error": "math_error", "expression": "<expr>"}`            |

---

### 4. `local_file_reader`

**Description:** Reads the contents of a local file by path and returns it as a string. The
readable file system is **sandboxed** to the `./data/` directory — paths outside this directory
are rejected. Used by agents when the user's query references a local document.

**Input Schema:**

```json
{
  "file_path": {
    "type": "string",
    "description": "Relative path to the file inside the ./data/ directory. Example: 'report.txt' or 'subdir/notes.md'. Do not include '../' or absolute paths.",
    "required": true
  },
  "encoding": {
    "type": "string",
    "description": "File encoding. Defaults to 'utf-8'.",
    "required": false,
    "default": "utf-8"
  }
}
```

**Output Schema:**

```json
{
  "file_path": "string",
  "content": "string",
  "size_bytes": "integer",
  "encoding": "string"
}
```

**Error Behavior:**

| Condition                         | Behavior                                                            |
|-----------------------------------|---------------------------------------------------------------------|
| File not found                    | Returns `{"error": "file_not_found", "file_path": "<path>"}`       |
| Path escape attempt (`../`)       | Returns `{"error": "access_denied", "file_path": "<path>"}`        |
| Absolute path provided            | Returns `{"error": "access_denied", "file_path": "<path>"}`        |
| File too large (> 1 MB)           | Returns `{"error": "file_too_large", "size_bytes": N}`              |
| Encoding error                    | Returns `{"error": "encoding_error", "encoding": "<enc>"}`         |

---

## Summary Table

| Tool name            | Agent(s) that use it       | External dependency      |
|----------------------|----------------------------|--------------------------|
| `duckduckgo_search`  | Research Agent             | `duckduckgo-search` pkg  |
| `wikipedia_lookup`   | Research Agent             | `wikipedia` pkg          |
| `python_calculator`  | Calculator Agent           | stdlib only (sandboxed)  |
| `local_file_reader`  | Any agent (routed by Supervisor) | filesystem (sandboxed) |

---

## Design Notes

- All tool inputs and outputs are validated with **Pydantic v2** models before the MCP server
  processes them.
- Agents call tools through `src/mcp_client.py` — they never import from `src/mcp_server/`
  directly. This is the **decoupling guarantee**.
- Error responses always use a consistent `{"error": "<error_code>", ...}` shape so the
  Supervisor agent can detect and handle failures uniformly.
- No tool performs any side effects other than `local_file_reader` reading from disk (read-only).