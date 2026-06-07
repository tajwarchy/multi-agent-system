"""
src/llm.py
LLM initialisation — the single place in the codebase that touches the
LLM provider. Every agent imports `get_llm()` from here.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 TO SWAP LLM PROVIDER — change ONLY the block inside get_llm():
 ──────────────────────────────────────────────────────────────
 Ollama (default, free, local):
     from langchain_ollama import ChatOllama
     return ChatOllama(model=model, temperature=temperature, ...)

 OpenAI:
     from langchain_openai import ChatOpenAI
     return ChatOpenAI(model="gpt-4o", temperature=temperature)

 Anthropic:
     from langchain_anthropic import ChatAnthropic
     return ChatAnthropic(model="claude-sonnet-4-5", temperature=temperature)

 Cohere:
     from langchain_cohere import ChatCohere
     return ChatCohere(model="command-r-plus", temperature=temperature)

 The graph, all agents, and all tools are 100% provider-agnostic.
 Only this file changes. No other file needs to be touched.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel

from src.config import load_config


@lru_cache(maxsize=1)
def get_llm() -> BaseChatModel:
    """
    Return a cached LLM instance configured from config/config.yaml.
    Called by every agent node — the instance is shared (thread-safe for reads).
    """
    cfg         = load_config()
    llm_cfg     = cfg["llm"]
    provider    = llm_cfg["provider"]
    model       = llm_cfg["model"]
    temperature = float(llm_cfg.get("temperature", 0.0))

    if provider == "ollama":
        # ── Default: Ollama local inference (M1 Metal via num_gpu=1) ──────────
        from langchain_ollama import ChatOllama
        return ChatOllama(
            model=model,
            temperature=temperature,
            num_predict=int(llm_cfg.get("num_predict", 1024)),
            base_url=llm_cfg.get("base_url", "http://localhost:11434"),
        )

    if provider == "openai":
        # ── Swap: OpenAI ──────────────────────────────────────────────────────
        # pip install langchain-openai
        # Set OPENAI_API_KEY in environment or .env
        from langchain_openai import ChatOpenAI  # type: ignore
        return ChatOpenAI(model=model, temperature=temperature)

    if provider == "anthropic":
        # ── Swap: Anthropic ───────────────────────────────────────────────────
        # pip install langchain-anthropic
        # Set ANTHROPIC_API_KEY in environment or .env
        from langchain_anthropic import ChatAnthropic  # type: ignore
        return ChatAnthropic(model=model, temperature=temperature)  # type: ignore[call-arg]

    raise ValueError(
        f"Unknown LLM provider '{provider}'. "
        "Supported: 'ollama', 'openai', 'anthropic'. "
        "Update config/config.yaml → llm.provider."
    )