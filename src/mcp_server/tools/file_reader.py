"""
src/mcp_server/tools/file_reader.py
Sandboxed local file reader tool — exposed via MCP.
Contract defined in docs/mcp_spec.md § local_file_reader.

Security model:
  - All paths resolved relative to the configured data_dir (./data/).
  - Path traversal attempts (../) are rejected before resolution.
  - Absolute paths are rejected.
  - Files larger than max_file_size_bytes are rejected.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator

from src.config import get


# ── Helpers ───────────────────────────────────────────────────────────────────

def _data_dir() -> Path:
    raw: str = get("mcp_server", "tools", "local_file_reader", "data_dir") or "./data"
    return Path(raw).resolve()

def _max_bytes() -> int:
    return int(get("mcp_server", "tools", "local_file_reader", "max_file_size_bytes") or 1_048_576)


# ── Input / Output schemas ────────────────────────────────────────────────────

class FileInput(BaseModel):
    file_path: str = Field(..., description="Relative path inside ./data/ directory.")
    encoding: str  = Field(default="utf-8", description="File encoding.")

    @field_validator("file_path")
    @classmethod
    def reject_dangerous_paths(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("file_path must not be empty")
        p = Path(v)
        if p.is_absolute():
            raise ValueError("absolute paths are not allowed")
        if ".." in p.parts:
            raise ValueError("path traversal ('..') is not allowed")
        return v.strip()


class FileOutput(BaseModel):
    file_path: str
    content: str
    size_bytes: int
    encoding: str


# ── Tool implementation ───────────────────────────────────────────────────────

def local_file_reader(file_path: str, encoding: str = "utf-8") -> dict[str, Any]:
    """
    Read a file from the sandboxed ./data/ directory.
    Returns a dict matching FileOutput schema (or an error dict).
    """
    try:
        inp = FileInput(file_path=file_path, encoding=encoding)
    except ValueError as e:
        err = str(e)
        if "traversal" in err or "absolute" in err:
            return {"error": "access_denied", "file_path": file_path}
        return {"error": "validation_error", "detail": err}

    data_dir   = _data_dir()
    max_bytes  = _max_bytes()
    full_path  = (data_dir / inp.file_path).resolve()

    # Double-check resolved path is still inside data_dir
    try:
        full_path.relative_to(data_dir)
    except ValueError:
        return {"error": "access_denied", "file_path": inp.file_path}

    if not full_path.exists():
        return {"error": "file_not_found", "file_path": inp.file_path}

    size = full_path.stat().st_size
    if size > max_bytes:
        return {"error": "file_too_large", "file_path": inp.file_path, "size_bytes": size}

    try:
        content = full_path.read_text(encoding=inp.encoding)
    except (UnicodeDecodeError, LookupError):
        return {"error": "encoding_error", "file_path": inp.file_path, "encoding": inp.encoding}
    except OSError as e:
        return {"error": "read_error", "file_path": inp.file_path, "detail": str(e)}

    out = FileOutput(
        file_path=inp.file_path,
        content=content,
        size_bytes=size,
        encoding=inp.encoding,
    )
    return out.model_dump()