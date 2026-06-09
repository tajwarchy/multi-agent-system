"""
src/api/main.py
FastAPI application — entry point for the multi-agent system REST API.

Endpoints:
  POST /run                 → run the agent graph, return answer + trace
  GET  /trace/{query_id}    → full step-by-step trace from SQLite
  GET  /traces              → list all recorded query IDs
  GET  /health              → system health check
  GET  /docs                → Swagger UI (auto-generated)
  GET  /redoc               → ReDoc UI (auto-generated)
"""

from __future__ import annotations

import logging
import time

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.api.models import HealthResponse
from src.api.routes.run   import router as run_router
from src.api.routes.trace import router as trace_router
from src.config import load_config

# ── Logging ───────────────────────────────────────────────────────────────────
cfg     = load_config()
api_cfg = cfg["fastapi"]
log_cfg = cfg["logging"]

logging.basicConfig(
    level   = getattr(logging, log_cfg["level"], logging.INFO),
    format  = log_cfg["format"],
)
log = logging.getLogger(__name__)

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title       = api_cfg["title"],
    version     = api_cfg["version"],
    description = api_cfg["description"],
    docs_url    = "/docs",
    redoc_url   = "/redoc",
)

# ── CORS (permissive for local dev — tighten for production) ──────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins  = ["*"],
    allow_methods  = ["*"],
    allow_headers  = ["*"],
)

# ── Request latency logging middleware ────────────────────────────────────────
@app.middleware("http")
async def log_requests(request: Request, call_next):
    t0       = time.monotonic()
    response = await call_next(request)
    ms       = (time.monotonic() - t0) * 1000
    log.info(
        "[API] %s %s → %d  (%.0f ms)",
        request.method, request.url.path, response.status_code, ms,
    )
    return response

# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(run_router)
app.include_router(trace_router)

# ── Health endpoint ───────────────────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse, summary="System health check")
def health() -> HealthResponse:
    chroma_docs = 0
    sqlite_runs = 0

    try:
        from src.memory.chroma_store import _get_collection
        chroma_docs = _get_collection().count()
    except Exception:
        pass

    try:
        from src.logging.sqlite_logger import get_all_query_ids
        sqlite_runs = len(get_all_query_ids())
    except Exception:
        pass

    return HealthResponse(
        status      = "ok",
        chroma_docs = chroma_docs,
        sqlite_runs = sqlite_runs,
    )

# ── Root ──────────────────────────────────────────────────────────────────────
@app.get("/", include_in_schema=False)
def root():
    return {
        "service":   api_cfg["title"],
        "version":   api_cfg["version"],
        "endpoints": ["/run", "/trace/{query_id}", "/traces", "/health", "/docs"],
    }

# ── Unhandled exception handler ───────────────────────────────────────────────
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    log.exception("[API] Unhandled exception on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code = 500,
        content     = {"detail": f"Internal server error: {exc}"},
    )