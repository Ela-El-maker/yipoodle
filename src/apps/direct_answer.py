from __future__ import annotations

import ast
import re
from dataclasses import dataclass


@dataclass(frozen=True)
class DirectAnswerResult:
    used: bool
    answer_type: str
    value: str | None


_ALLOWED_BINOPS = (ast.Add, ast.Sub, ast.Mult, ast.Div)
_ALLOWED_UNARY = (ast.UAdd, ast.USub)


def _safe_eval(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, _ALLOWED_UNARY):
        val = _safe_eval(node.operand)
        return val if isinstance(node.op, ast.UAdd) else -val
    if isinstance(node, ast.BinOp) and isinstance(node.op, _ALLOWED_BINOPS):
        left = _safe_eval(node.left)
        right = _safe_eval(node.right)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if right == 0:
            raise ZeroDivisionError("division by zero")
        return left / right
    raise ValueError("unsupported expression")


def _is_math_candidate(question: str) -> bool:
    q = (question or "").strip().lower()
    if not q:
        return False
    if re.search(r"\d", q) is None:
        return False
    if re.fullmatch(r"[\d\s\+\-\*\/\(\)\.=\?]+", q) is None:
        return False
    return True


def try_direct_answer(question: str, max_complexity: int = 2) -> DirectAnswerResult:
    if not _is_math_candidate(question):
        return DirectAnswerResult(used=False, answer_type="none", value=None)

    expr = (question or "").strip()
    expr = expr.replace("=", " ").replace("?", " ").strip()
    # keep only arithmetic characters
    expr = re.sub(r"[^\d\+\-\*\/\(\)\.\s]", "", expr)
    if not expr:
        return DirectAnswerResult(used=False, answer_type="none", value=None)

    op_count = len(re.findall(r"[\+\-\*\/]", expr))
    if op_count > int(max_complexity):
        return DirectAnswerResult(used=False, answer_type="none", value=None)

    try:
        tree = ast.parse(expr, mode="eval")
        val = _safe_eval(tree)
    except Exception:
        return DirectAnswerResult(used=False, answer_type="none", value=None)

    if abs(val - round(val)) < 1e-9:
        out = str(int(round(val)))
    else:
        out = (f"{val:.8f}").rstrip("0").rstrip(".")
    return DirectAnswerResult(used=True, answer_type="arithmetic", value=out)
