"""
src/memory/chroma_store.py
ChromaDB agent memory — persistent, local, semantic search over past traces.

Two public functions used by the graph:
  - retrieve_similar_traces(query)  → called by supervisor before routing
  - persist_trace(state)            → called by summarizer after completion

Storage layout:
  - Each document = one completed query's full reasoning trace (JSON)
  - Each document's ID = query_id
  - Metadata: query, route, timestamp, had_errors
  - Embedding: ChromaDB default (sentence-transformers/all-MiniLM-L6-v2)
    via chromadb's built-in embedding function (no Ollama needed for embeddings)
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

import chromadb
from chromadb.utils import embedding_functions

from src.config import load_config
from src.graph.state import AgentState

log = logging.getLogger(__name__)
cfg = load_config()["chromadb"]


# ── ChromaDB client (singleton) ───────────────────────────────────────────────

@lru_cache(maxsize=1)
def _get_client() -> chromadb.PersistentClient:
    persist_dir = Path(cfg["persist_directory"])
    persist_dir.mkdir(parents=True, exist_ok=True)
    log.info("[ChromaDB] persist_directory=%s", persist_dir.resolve())
    return chromadb.PersistentClient(path=str(persist_dir))


@lru_cache(maxsize=1)
def _get_collection() -> chromadb.Collection:
    client = _get_client()
    # Use ChromaDB's built-in sentence-transformer embedding function
    # (downloads ~90MB model on first run — cached locally after that)
    ef = embedding_functions.DefaultEmbeddingFunction()
    collection = client.get_or_create_collection(
        name=cfg["collection_name"],
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )
    log.info(
        "[ChromaDB] collection '%s' ready (%d documents)",
        cfg["collection_name"],
        collection.count(),
    )
    return collection


# ── Serialise a trace for storage ─────────────────────────────────────────────

def _serialise_trace(state: AgentState) -> str:
    """
    Convert a completed AgentState into a compact string for embedding.
    We embed the query + route + final_answer + tool calls so retrieval
    is semantically meaningful.
    """
    parts = [
        f"Query: {state.get('query', '')}",
        f"Route: {state.get('route', '')}",
        f"Final answer: {state.get('final_answer', '')}",
    ]

    for step in state.get("agent_trace", []):
        tool = step.get("tool_called") or "none"
        parts.append(
            f"Agent {step['agent_name']} used tool={tool} "
            f"success={step['success']}"
        )

    if state.get("errors"):
        parts.append(f"Errors: {'; '.join(state['errors'])}")

    return "\n".join(parts)


# ── Public API ────────────────────────────────────────────────────────────────

def persist_trace(state: AgentState) -> None:
    """
    Embed and store the completed reasoning trace in ChromaDB.
    Called by summarizer_node after every successful graph run.
    Non-fatal — errors are logged but never raised.
    """
    try:
        query_id = state.get("query_id", "unknown")
        doc      = _serialise_trace(state)

        metadata: dict[str, Any] = {
            "query":      state.get("query", "")[:500],
            "route":      state.get("route", ""),
            "timestamp":  datetime.now(timezone.utc).isoformat(),
            "had_errors": len(state.get("errors", [])) > 0,
            "answer":     (state.get("final_answer") or "")[:500],
        }

        collection = _get_collection()
        # upsert — idempotent if the same query_id is stored twice
        collection.upsert(
            ids=[query_id],
            documents=[doc],
            metadatas=[metadata],
        )
        log.info("[ChromaDB] persisted trace query_id=%s  total=%d", query_id, collection.count())

    except Exception as e:
        log.error("[ChromaDB] persist_trace failed (non-fatal): %s", e)


def retrieve_similar_traces(query: str) -> str:
    """
    Query ChromaDB for the top-k most semantically similar past traces.
    Returns a formatted string injected into the supervisor's context,
    or an empty string if no relevant traces exist yet.
    """
    try:
        collection = _get_collection()

        if collection.count() == 0:
            return ""

        n_results  = min(int(cfg["n_results"]), collection.count())
        threshold  = float(cfg["similarity_threshold"])

        results = collection.query(
            query_texts=[query],
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )

        docs      = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances",  [[]])[0]

        # ChromaDB cosine distance: 0 = identical, 2 = opposite
        # Convert to similarity: 1 - (distance / 2)
        relevant = []
        for doc, meta, dist in zip(docs, metadatas, distances):
            similarity = 1.0 - (dist / 2.0)
            if similarity >= threshold:
                relevant.append((similarity, doc, meta))

        if not relevant:
            return ""

        parts = ["=== Relevant past reasoning traces ==="]
        for sim, doc, meta in relevant:
            parts.append(
                f"\n[similarity={sim:.2f}  route={meta.get('route')}  "
                f"timestamp={meta.get('timestamp', '')[:19]}]\n{doc}"
            )

        context = "\n".join(parts)
        log.info("[ChromaDB] retrieved %d relevant traces for query", len(relevant))
        return context

    except Exception as e:
        log.error("[ChromaDB] retrieve_similar_traces failed (non-fatal): %s", e)
        return ""