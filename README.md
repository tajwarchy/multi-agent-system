# Multi-Agent System with LangGraph, MCP, and Local Tool Orchestration

A production-grade multi-agent AI system built with LangGraph, the Model Context Protocol (MCP), ChromaDB memory, SQLite observability, and a FastAPI REST layer — all running locally on Apple Silicon with zero API costs.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Architecture Overview](#architecture-overview)
3. [Agent Graph State Machine](#agent-graph-state-machine)
4. [System Design Deep Dive](#system-design-deep-dive)
5. [MCP Tool Contract](#mcp-tool-contract)
6. [API Reference](#api-reference)
7. [Configuration](#configuration)
8. [LLM Provider Swap](#llm-provider-swap)
9. [Running Tests](#running-tests)
10. [Docker Deployment](#docker-deployment)
11. [Project Structure](#project-structure)

---

## Quick Start

### Prerequisites

- macOS (Apple Silicon M1/M2/M3)
- [Conda](https://docs.conda.io/en/latest/miniconda.html)
- [Ollama](https://ollama.com) installed and running
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) (for containerised deployment)

### Local setup

```bash
# 1. Clone the repo
git clone https://github.com/tajwarchy/multi-agent-system.git
cd multi-agent-system

# 2. Create and activate the conda environment
conda env create -f environment.yml
conda activate multi_agent_sys

# 3. Pull the LLM
ollama pull mistral

# 4. Verify setup
python scripts/verify_setup.py

# 5. Run a query end-to-end
python -m scripts.run_query "What is the square root of 256?"
```

### Start the API server

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

Open `http://localhost:8000/docs` for the interactive Swagger UI.

### Docker (one command)

```bash
# Ollama must be running on the host first
ollama serve

docker compose up --build
```

---

## Architecture Overview

```
User Query
    │
    ▼
FastAPI  POST /run
    │
    ▼
┌─────────────────────────────────────────────────────┐
│                  LangGraph Graph                     │
│                                                      │
│  ┌─────────────┐                                     │
│  │  Supervisor │◄── ChromaDB (memory recall)         │
│  │   Agent     │                                     │
│  └──────┬──────┘                                     │
│         │ conditional routing                        │
│    ┌────┴─────────────────┐                          │
│    │                      │                          │
│  ┌─┴──────────┐  ┌────────┴───────┐                  │
│  │  Research  │  │  Calculator    │  (parallel)      │
│  │   Agent    │  │   Agent        │                  │
│  └─────┬──────┘  └───────┬────────┘                  │
│        │                 │                           │
│        └────────┬────────┘                           │
│                 │                                    │
│         ┌───────┴──────┐                             │
│         │  Summarizer  │                             │
│         │    Agent     │──► ChromaDB (persist)       │
│         └──────────────┘                             │
│                                                      │
│  Every node  ──► SQLite (step logger)                │
└─────────────────────────────────────────────────────┘
         │
         ▼
   MCP Server (subprocess, stdio)
   ├── duckduckgo_search
   ├── wikipedia_lookup
   ├── python_calculator
   └── local_file_reader
```

**Key design principle:** agents never import tools directly. Every tool call goes through the MCP client → MCP server boundary. The graph, agents, and tools are three completely decoupled layers.

---

## Agent Graph State Machine

Every node, every conditional edge, every terminal state:

```
START
  │
  ▼
[supervisor]  ── reads ChromaDB memory context
  │
  ├─ route="research_only"    ──► [research]    ──► [summarizer] ──► END
  ├─ route="calculation_only" ──► [calculator]  ──► [summarizer] ──► END
  ├─ route="both"             ──► [both_research] (research + calculator)
  │                                                ──► [summarizer] ──► END
  └─ route="summarize_only"   ──────────────────► [summarizer] ──► END

Failure path (any node):
  [node fails] ──► error appended to state["errors"]
               ──► retry once (max_retries=1)
               ──► partial output passed to [summarizer]
               ──► [summarizer] always runs, never skipped
               ──► graph never raises an unhandled exception
```

**Shared state** (`AgentState` TypedDict) flows through every node:

| Field | Set by | Read by |
|---|---|---|
| `query`, `query_id` | `run_graph()` entry | all nodes |
| `route` | supervisor | graph router |
| `memory_context` | supervisor (ChromaDB) | summarizer |
| `research_output` | research agent | calculator (both path), summarizer |
| `calculation_output` | calculator agent | summarizer |
| `final_answer` | summarizer | API response |
| `agent_trace` | every node (appended) | SQLite logger, API |
| `errors` | any failing node | summarizer, API |

---

## System Design Deep Dive

### Why LangGraph over a simple loop?

A naive implementation would chain agents in a `for` loop: supervisor → research → calculator → summarizer. This breaks immediately when you need:

- **Conditional routing**: not every query needs all agents. A simple loop runs everything every time, wasting latency and compute.
- **Parallel execution**: research and calculator are independent for `both` queries. A loop is sequential by definition.
- **Cycles and retries**: a loop can't re-enter a previous step. LangGraph nodes can be re-invoked by routing back to them.
- **Shared mutable state**: LangGraph's `StateGraph` passes a single typed state object between nodes and handles merging partial updates. A loop requires manually threading state through every function call.

LangGraph models the system as a **directed graph** — nodes are agents, edges are control flow, conditional edges are routing decisions. This is the same mental model as a state machine, which makes reasoning about failure modes, retries, and parallelism tractable.

### What is MCP and why does it matter?

The Model Context Protocol is an open standard for how AI agents discover and call tools. In this system:

- The **MCP server** owns tool implementations. It exposes a typed schema for each tool and handles execution.
- **Agents** call tools through the **MCP client** — a thin wrapper that speaks the protocol. Agents never import tool code directly.

This decoupling means:
- A tool can be rewritten (e.g. swap DuckDuckGo for Brave Search) without touching any agent code.
- New tools can be added to the MCP server and immediately become available to all agents.
- The MCP server can be moved to a remote host, a different process, or replaced entirely — the agent interface doesn't change.
- This maps directly to the **interface segregation** principle in software design: agents depend on the MCP contract, not on implementations.

### The supervisor pattern and microservices mapping

The supervisor agent is a **centralised orchestrator**: it receives every query, classifies it, and delegates to specialist sub-agents. This maps directly to an API gateway pattern in microservices:

| Multi-agent concept | Microservices equivalent |
|---|---|
| Supervisor agent | API gateway / router |
| Research agent | Search microservice |
| Calculator agent | Computation microservice |
| Summarizer agent | Aggregation / BFF service |
| MCP server | Internal service mesh |
| AgentState | Request context / correlation ID |

**Tradeoffs of a centralised orchestrator:**
- ✅ Single point of routing logic — easy to reason about, test, and modify.
- ✅ Easy to add new agents — just add a new routing label and node.
- ❌ Single point of failure — if the supervisor crashes, nothing runs. Mitigated here by wrapping the supervisor in the graph's global exception handler.
- ❌ Bottleneck — every query passes through the supervisor. At scale, this becomes a latency constraint.

### Shared state management — risks and safeguards

In a multi-agent graph, all agents read from and write to a single `AgentState` TypedDict. Risks:

1. **One agent's bad output corrupts downstream agents.** Mitigation: every field is `Optional`. Summarizer checks for `None` before using any sub-agent output. A failed agent writes an error to `state["errors"]` rather than writing malformed data to its output field.

2. **Concurrent writes (both path) cause race conditions.** Mitigation: the `both_research` node runs research and calculator sequentially in the current implementation (single Ollama instance serialises LLM calls anyway). A future async implementation using `asyncio.gather()` would require explicit merge logic — documented in `graph.py`.

3. **State grows unboundedly across a long session.** Mitigation: `agent_trace` entries truncate output to 500 chars. SQLite is the persistent store for full outputs — state is ephemeral per graph run.

### Scalability ceiling and what breaks first

**Current system serves one request at a time.** The bottleneck is the single Ollama instance — it processes one LLM inference at a time (no batching, no parallelism). Under 200 concurrent users:

| Component | Behaviour under load |
|---|---|
| FastAPI | Handles concurrency natively (async I/O, multiple workers) |
| MCP server | Spawned per-request as a subprocess — scales with workers |
| ChromaDB | Local persistent client — not thread-safe for concurrent writes |
| SQLite | File-based — concurrent writes will queue/lock |
| **Ollama** | **Single instance — all requests queue. This breaks first.** |

**What you'd add for production scale:**
- Multiple Ollama instances behind a load balancer (e.g. Nginx), or swap to a scalable API provider (OpenAI, Anthropic) by changing one line in `config.yaml`.
- Replace local ChromaDB with a hosted vector DB (Pinecone, Weaviate) with connection pooling.
- Replace SQLite with PostgreSQL.
- Run FastAPI with multiple Uvicorn workers: `uvicorn src.api.main:app --workers 4`.

### LLM provider abstraction

Because LangGraph uses LangChain's `BaseChatModel` abstraction, the entire system is provider-agnostic. The graph, all agent nodes, and all tool calls are identical regardless of which LLM is used. **Only one file changes: `src/llm.py`, and only one block inside it.**

---

## MCP Tool Contract

Full specification in `docs/mcp_spec.md`. Summary:

| Tool | Agent | Input | Output |
|---|---|---|---|
| `duckduckgo_search` | Research | `query`, `max_results` | `results[]`, `result_count` |
| `wikipedia_lookup` | Research | `topic`, `sentences` | `title`, `summary`, `url` |
| `python_calculator` | Calculator | `expression` | `result`, `result_type` |
| `local_file_reader` | Any | `file_path`, `encoding` | `content`, `size_bytes` |

All tools return `{"error": "<code>", ...}` on failure — never raise exceptions to the caller.

---

## API Reference

### `POST /run`

Run the multi-agent graph.

**Request:**
```json
{ "query": "What is the GDP of Germany and what is 5% of it?" }
```

**Response:**
```json
{
  "query_id":     "abc-123",
  "query":        "What is the GDP...",
  "route":        "both",
  "final_answer": "Germany's GDP is approximately $4.3 trillion...",
  "agent_trace":  [ { "agent_name": "supervisor", "latency_ms": 3200, "success": true }, ... ],
  "errors":       [],
  "total_steps":  4
}
```

### `GET /trace/{query_id}`

Fetch the full step-by-step agent trace from SQLite.

**Response:**
```json
{
  "query_id":    "abc-123",
  "total_steps": 4,
  "steps": [
    { "agent_name": "supervisor", "tool_called": null, "latency_ms": 3200, "success": true, ... },
    { "agent_name": "research",   "tool_called": "duckduckgo_search", ... },
    ...
  ]
}
```

### `GET /traces`

List all recorded query IDs, newest first.

### `GET /health`

```json
{ "status": "ok", "chroma_docs": 12, "sqlite_runs": 12 }
```

---

## Configuration

All parameters live in `config/config.yaml`. Nothing is hardcoded.

Key sections:

```yaml
llm:
  provider: ollama      # swap to "openai" or "anthropic" — see below
  model: mistral

graph:
  max_retries: 1        # sub-agent retry count on failure
  recursion_limit: 25   # LangGraph max node traversals

chromadb:
  n_results: 3          # top-k past traces injected as memory context
  similarity_threshold: 0.75

sqlite:
  db_path: ./storage/logs/agent_steps.db
```

---

## LLM Provider Swap

**This entire system works with any LangChain-supported LLM. Only one line changes.**

Open `src/llm.py` and change the provider block:

```python
# Current (free, local):
from langchain_ollama import ChatOllama
return ChatOllama(model="mistral", ...)

# Swap to OpenAI:
from langchain_openai import ChatOpenAI
return ChatOpenAI(model="gpt-4o", temperature=0)

# Swap to Anthropic:
from langchain_anthropic import ChatAnthropic
return ChatAnthropic(model="claude-sonnet-4-5", temperature=0)
```

Or change `config/config.yaml`:
```yaml
llm:
  provider: openai   # was: ollama
  model: gpt-4o
```

The graph, all agents, and all tools are completely provider-agnostic.

---

## Running Tests

```bash
# All tests (excluding network)
pytest tests/ -v -m "not network"

# Individual suites
pytest tests/test_tools.py -v -m "not network"    # MCP tool unit tests
pytest tests/test_mcp_server.py -v -m "not network" # MCP integration tests
pytest tests/test_graph.py -v                       # Graph + node tests
pytest tests/test_memory_logging.py -v              # ChromaDB + SQLite tests
pytest tests/test_api.py -v                         # FastAPI endpoint tests

# Network tests (requires internet, may hit rate limits)
pytest tests/ -v -m network
```

---

## Docker Deployment

```bash
# Prerequisites: Ollama running on host
ollama serve
ollama pull mistral

# Build and start
docker compose up --build

# Verify
curl http://localhost:8000/health

# Stop
docker compose down
```

The container calls Ollama on the host via `host.docker.internal:11434`. Storage (ChromaDB + SQLite) is persisted via a volume mount at `./storage/`.

---

## Project Structure

```
multi-agent-system/
├── config/config.yaml          # Central config — all parameters
├── docs/
│   ├── agent_graph.png         # State machine diagram
│   └── mcp_spec.md             # MCP tool contract
├── src/
│   ├── config.py               # Config loader
│   ├── llm.py                  # LLM abstraction (swap point)
│   ├── mcp_client.py           # MCP client wrapper
│   ├── mcp_server/             # MCP server + 4 tools
│   ├── graph/                  # LangGraph agents + graph
│   ├── memory/                 # ChromaDB store
│   ├── logging/                # SQLite logger
│   └── api/                    # FastAPI app + routes
├── tests/                      # Full test suite
├── scripts/                    # CLI tools
├── data/                       # Sandboxed file reader directory
├── storage/                    # Runtime: ChromaDB + SQLite (gitignored)
├── Dockerfile.api
├── docker-compose.yml
├── environment.yml             # Conda env
└── requirements.txt            # Pip env (Docker)
```