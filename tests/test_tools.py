"""
tests/test_tools.py
Unit tests for each MCP tool implementation.
These call the Python functions directly — no MCP server needed.

Run:
    pytest tests/test_tools.py -v
"""

import os
import tempfile
from pathlib import Path

import pytest

from src.mcp_server.tools.calculator  import python_calculator
from src.mcp_server.tools.file_reader import local_file_reader
from src.mcp_server.tools.search      import duckduckgo_search
from src.mcp_server.tools.wikipedia   import wikipedia_lookup


# ═══════════════════════════════════════════════════════════════════════════════
# Calculator
# ═══════════════════════════════════════════════════════════════════════════════

class TestCalculator:

    def test_basic_arithmetic(self):
        r = python_calculator("2 + 2")
        assert r["result"] == 4.0
        assert "error" not in r

    def test_multiplication(self):
        r = python_calculator("6 * 7")
        assert r["result"] == 42.0

    def test_exponent(self):
        r = python_calculator("2 ** 10")
        assert r["result"] == 1024.0

    def test_sqrt(self):
        r = python_calculator("sqrt(144)")
        assert r["result"] == 12.0

    def test_round(self):
        r = python_calculator("round(3.14159, 2)")
        assert r["result"] == 3.14

    def test_floor_ceil(self):
        assert python_calculator("floor(2.9)")["result"] == 2.0
        assert python_calculator("ceil(2.1)")["result"]  == 3.0

    def test_chained(self):
        r = python_calculator("sqrt(144) + round(3.14, 1)")
        assert r["result"] == pytest.approx(15.1)

    def test_pi_constant(self):
        r = python_calculator("round(pi, 5)")
        assert r["result"] == pytest.approx(3.14159)

    def test_division_by_zero(self):
        r = python_calculator("1 / 0")
        assert r["error"] == "division_by_zero"

    def test_syntax_error(self):
        r = python_calculator("2 +* 3")
        assert r["error"] == "syntax_error"

    def test_disallowed_import(self):
        r = python_calculator("__import__('os')")
        assert r["error"] in ("syntax_error", "disallowed_operation")

    def test_disallowed_assignment(self):
        r = python_calculator("x = 5")
        assert r["error"] in ("syntax_error", "disallowed_operation")

    def test_empty_expression(self):
        r = python_calculator("")
        assert "error" in r

    def test_log(self):
        r = python_calculator("round(log(e), 4)")
        assert r["result"] == 1.0


# ═══════════════════════════════════════════════════════════════════════════════
# File reader
# ═══════════════════════════════════════════════════════════════════════════════

class TestFileReader:
    """
    These tests write temp files into ./data/ so the tool can find them.
    The data dir is created if it doesn't exist.
    """

    @pytest.fixture(autouse=True)
    def ensure_data_dir(self):
        Path("data").mkdir(exist_ok=True)

    @pytest.fixture
    def sample_file(self):
        p = Path("data/test_sample.txt")
        p.write_text("Hello from the test file.", encoding="utf-8")
        yield "test_sample.txt"
        p.unlink(missing_ok=True)

    def test_read_existing_file(self, sample_file):
        r = local_file_reader(sample_file)
        assert r["content"] == "Hello from the test file."
        assert r["size_bytes"] > 0
        assert "error" not in r

    def test_file_not_found(self):
        r = local_file_reader("does_not_exist.txt")
        assert r["error"] == "file_not_found"

    def test_path_traversal_rejected(self):
        r = local_file_reader("../config/config.yaml")
        assert r["error"] == "access_denied"

    def test_absolute_path_rejected(self):
        r = local_file_reader("/etc/passwd")
        assert r["error"] == "access_denied"

    def test_empty_path_rejected(self):
        r = local_file_reader("")
        assert "error" in r

    def test_subdir_file(self):
        sub = Path("data/subdir")
        sub.mkdir(exist_ok=True)
        p = sub / "note.txt"
        p.write_text("nested file", encoding="utf-8")
        r = local_file_reader("subdir/note.txt")
        assert r["content"] == "nested file"
        p.unlink()
        sub.rmdir()


# ═══════════════════════════════════════════════════════════════════════════════
# DuckDuckGo search  (network — skip in CI with no internet)
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.network
class TestDuckDuckGoSearch:

    def test_basic_search_returns_results(self):
        r = duckduckgo_search("Python programming language")
        if r.get("error") == "network_error":
            pytest.skip("DuckDuckGo rate-limited — transient, not a code bug")
        assert r["result_count"] > 0
        assert len(r["results"]) > 0
        first = r["results"][0]
        assert "title" in first and "snippet" in first and "url" in first

    def test_max_results_respected(self):
        r = duckduckgo_search("Python", max_results=3)
        assert r["result_count"] <= 3

    def test_empty_query_rejected(self):
        r = duckduckgo_search("")
        assert "error" in r

    def test_query_echoed(self):
        r = duckduckgo_search("LangGraph tutorial")
        if "error" not in r:
            assert r["query"] == "LangGraph tutorial"


# ═══════════════════════════════════════════════════════════════════════════════
# Wikipedia  (network — skip in CI with no internet)
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.network
class TestWikipediaLookup:

    def test_known_topic(self):
        r = wikipedia_lookup("Python (programming language)")
        if r.get("error") == "network_error":
            pytest.skip("Wikipedia rate-limited — transient, not a code bug")
        assert len(r["summary"]) > 0
        assert r["url"].startswith("http")

    def test_sentences_count(self):
        r = wikipedia_lookup("Albert Einstein", sentences=3)
        if r.get("error") == "network_error":
            pytest.skip("Wikipedia rate-limited — transient, not a code bug")
        assert r["sentences_returned"] == 3

    def test_page_not_found(self):
        r = wikipedia_lookup("xkq3h9f2nonsenseterm99887766")
        if r.get("error") == "network_error":
            pytest.skip("Wikipedia rate-limited — transient, not a code bug")
        assert r.get("error") in ("page_not_found", "disambiguation")

    def test_empty_topic_rejected(self):
        r = wikipedia_lookup("")
        assert "error" in r