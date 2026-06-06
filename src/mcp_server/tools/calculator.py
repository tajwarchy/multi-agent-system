"""
src/mcp_server/tools/calculator.py
Sandboxed Python calculator tool — exposed via MCP.
Contract defined in docs/mcp_spec.md § python_calculator.

Security model:
  - Whitelist of allowed math names only (no builtins, no imports).
  - AST is parsed and every node type is checked before eval.
  - No assignment, no function definitions, no attribute access.
"""

from __future__ import annotations

import ast
import math
import operator
from typing import Any

from pydantic import BaseModel, Field, field_validator

from src.config import get


# ── Whitelist ─────────────────────────────────────────────────────────────────

_ALLOWED_FUNCTIONS: dict[str, Any] = {
    "sqrt":  math.sqrt,
    "abs":   abs,
    "round": round,
    "floor": math.floor,
    "ceil":  math.ceil,
    "pow":   math.pow,
    "log":   math.log,
    "log10": math.log10,
    "pi":    math.pi,
    "e":     math.e,
}

_ALLOWED_NODE_TYPES = {
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.Call,
    ast.Constant, ast.Name, ast.Load,   # ast.Load is the context on every Name node
    ast.keyword,                         # needed for round(x, ndigits=2) style calls
    # operators — exact Python 3.11 AST node names
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Mod,
    ast.Pow, ast.FloorDiv, ast.USub, ast.UAdd,
}


# ── AST safety check ─────────────────────────────────────────────────────────

def _safe_ast(node: ast.AST) -> None:
    """Recursively validate that every AST node is in the whitelist."""
    if type(node) not in _ALLOWED_NODE_TYPES:
        raise ValueError(f"Disallowed operation: {type(node).__name__}")
    if isinstance(node, ast.Name) and node.id not in _ALLOWED_FUNCTIONS:
        raise ValueError(f"Disallowed name: '{node.id}'")
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise ValueError("Disallowed call target")
        if node.func.id not in _ALLOWED_FUNCTIONS:
            raise ValueError(f"Disallowed function: '{node.func.id}'")
    for child in ast.iter_child_nodes(node):
        _safe_ast(child)


# ── Input / Output schemas ────────────────────────────────────────────────────

class CalcInput(BaseModel):
    expression: str = Field(..., description="Mathematical expression to evaluate.")

    @field_validator("expression")
    @classmethod
    def expr_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("expression must not be empty")
        return v.strip()


class CalcOutput(BaseModel):
    expression: str
    result: float
    result_type: str


# ── Tool implementation ───────────────────────────────────────────────────────

def python_calculator(expression: str) -> dict[str, Any]:
    """
    Safely evaluate a mathematical expression.
    Returns a dict matching CalcOutput schema (or an error dict).
    """
    try:
        inp = CalcInput(expression=expression)
    except ValueError as e:
        return {"error": "validation_error", "detail": str(e)}

    # Parse AST
    try:
        tree = ast.parse(inp.expression, mode="eval")
    except SyntaxError:
        return {"error": "syntax_error", "expression": inp.expression}

    # Safety check
    try:
        _safe_ast(tree)
    except ValueError as e:
        return {"error": "disallowed_operation", "expression": inp.expression, "detail": str(e)}

    # Evaluate
    try:
        result = eval(  # noqa: S307  (safe: AST-validated whitelist)
            compile(tree, filename="<calc>", mode="eval"),
            {"__builtins__": {}},
            _ALLOWED_FUNCTIONS,
        )
    except ZeroDivisionError:
        return {"error": "division_by_zero", "expression": inp.expression}
    except OverflowError:
        return {"error": "math_error", "expression": inp.expression, "detail": "overflow"}
    except Exception as e:
        return {"error": "math_error", "expression": inp.expression, "detail": str(e)}

    if not isinstance(result, (int, float)):
        return {"error": "math_error", "expression": inp.expression, "detail": "non-numeric result"}

    out = CalcOutput(
        expression=inp.expression,
        result=float(result),
        result_type=type(result).__name__,
    )
    return out.model_dump()