"""
src/config.py
Loads config/config.yaml once at import time and exposes a typed
settings object used everywhere in the codebase.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml


_CONFIG_PATH = Path(__file__).parent.parent / "config" / "config.yaml"


@lru_cache(maxsize=1)
def load_config() -> dict[str, Any]:
    """Load and cache config.yaml. Call this from anywhere."""
    with open(_CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


def get(section: str, *keys: str, default: Any = None) -> Any:
    """
    Convenience accessor.
    Example: get("llm", "model") → "mistral"
             get("mcp_server", "tools", "python_calculator", "enabled") → True
    """
    cfg = load_config()
    node: Any = cfg.get(section, {})
    for k in keys:
        if isinstance(node, dict):
            node = node.get(k, default)
        else:
            return default
    return node