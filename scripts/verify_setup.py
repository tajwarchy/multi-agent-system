"""
verify_setup.py
Run after `conda activate multi_agent_sys` to confirm the environment
and all dependencies are correctly installed.

Usage:
    python scripts/verify_setup.py
"""

import sys
import subprocess

# ── Minimum Python version ───────────────────────────────────────────────────
REQUIRED = (3, 11)
assert sys.version_info >= REQUIRED, (
    f"Python {REQUIRED[0]}.{REQUIRED[1]}+ required, got {sys.version}"
)

CHECKS = [
    # (import_name, display_name)
    ("langchain",            "LangChain"),
    ("langchain_core",       "LangChain Core"),
    ("langchain_community",  "LangChain Community"),
    ("langchain_ollama",     "LangChain Ollama"),
    ("langgraph",            "LangGraph"),
    ("mcp",                  "MCP Python SDK"),
    ("ollama",               "Ollama Python client"),
    ("duckduckgo_search",    "DuckDuckGo Search"),
    ("wikipedia",            "Wikipedia"),
    ("chromadb",             "ChromaDB"),
    ("sqlalchemy",           "SQLAlchemy"),
    ("aiosqlite",            "aiosqlite"),
    ("fastapi",              "FastAPI"),
    ("uvicorn",              "Uvicorn"),
    ("httpx",                "HTTPX"),
    ("pydantic",             "Pydantic"),
    ("pydantic_settings",    "Pydantic Settings"),
    ("yaml",                 "PyYAML"),
    ("dotenv",               "python-dotenv"),
    ("pytest",               "Pytest"),
    ("rich",                 "Rich"),
    ("tenacity",             "Tenacity"),
]

GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
RESET  = "\033[0m"
BOLD   = "\033[1m"

def check_imports() -> list[str]:
    failures = []
    print(f"\n{BOLD}── Package imports ─────────────────────────────────────────{RESET}")
    for module, name in CHECKS:
        try:
            __import__(module)
            print(f"  {GREEN}✓{RESET}  {name}")
        except ImportError as e:
            print(f"  {RED}✗{RESET}  {name}  ({e})")
            failures.append(name)
    return failures


def check_ollama() -> bool:
    print(f"\n{BOLD}── Ollama server ───────────────────────────────────────────{RESET}")
    try:
        import ollama
        models = ollama.list()
        names = [m.model for m in models.models]
        print(f"  {GREEN}✓{RESET}  Ollama running — models: {names if names else '(none pulled yet)'}")
        target = "mistral"
        if any(target in n for n in names):
            print(f"  {GREEN}✓{RESET}  '{target}' model available")
        else:
            print(
                f"  {YELLOW}!{RESET}  '{target}' not found. Pull it with:\n"
                f"         ollama pull {target}"
            )
        return True
    except Exception as e:
        print(f"  {RED}✗{RESET}  Cannot reach Ollama: {e}")
        print(
            f"         Start Ollama with: ollama serve\n"
            f"         Then pull the model: ollama pull mistral"
        )
        return False


def check_config() -> bool:
    print(f"\n{BOLD}── Config file ─────────────────────────────────────────────{RESET}")
    from pathlib import Path
    cfg = Path("config/config.yaml")
    if cfg.exists():
        print(f"  {GREEN}✓{RESET}  config/config.yaml found")
        return True
    else:
        print(f"  {RED}✗{RESET}  config/config.yaml not found (run from project root)")
        return False


def check_dirs() -> None:
    print(f"\n{BOLD}── Required directories ────────────────────────────────────{RESET}")
    from pathlib import Path
    dirs = [
        "src", "src/mcp_server", "src/mcp_server/tools",
        "src/graph", "src/graph/nodes", "src/memory",
        "src/logging", "src/api", "src/api/routes",
        "config", "docs", "data", "storage/chroma", "storage/logs", "tests",
    ]
    for d in dirs:
        p = Path(d)
        if p.exists():
            print(f"  {GREEN}✓{RESET}  {d}/")
        else:
            p.mkdir(parents=True, exist_ok=True)
            print(f"  {YELLOW}+{RESET}  {d}/  (created)")


def main() -> None:
    print(f"\n{BOLD}Multi-Agent System — Environment Verification{RESET}")
    print(f"Python {sys.version}")

    import_failures = check_imports()
    ollama_ok       = check_ollama()
    config_ok       = check_config()
    check_dirs()

    print(f"\n{BOLD}── Summary ─────────────────────────────────────────────────{RESET}")
    if not import_failures and ollama_ok and config_ok:
        print(f"  {GREEN}{BOLD}All checks passed. Ready for Phase 2.{RESET}\n")
        sys.exit(0)
    else:
        if import_failures:
            print(f"  {RED}✗  Missing packages: {', '.join(import_failures)}{RESET}")
            print(f"     Run: conda env update -f environment.yml --prune")
        if not ollama_ok:
            print(f"  {RED}✗  Ollama not reachable — start with: ollama serve{RESET}")
        if not config_ok:
            print(f"  {RED}✗  config/config.yaml missing{RESET}")
        print()
        sys.exit(1)


if __name__ == "__main__":
    main()