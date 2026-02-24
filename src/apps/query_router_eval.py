from __future__ import annotations

from pathlib import Path
from typing import Any
import json

from src.apps.query_router import load_router_config, route_query


_ALLOWED_MODES = {"ask", "research", "monitor", "notes"}


def _load_cases(path: str) -> list[dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"router eval cases not found: {path}")
    raw = json.loads(p.read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        raw = raw.get("cases", [])
    if not isinstance(raw, list):
        raise ValueError("router eval cases must be a list or an object with 'cases' list")

    out: list[dict[str, Any]] = []
    for i, row in enumerate(raw):
        if not isinstance(row, dict):
            raise ValueError(f"case {i} must be an object")
        q = str(row.get("question", "")).strip()
        exp = str(row.get("expected_mode", "")).strip().lower()
        if not q:
            raise ValueError(f"case {i} missing question")
        if exp not in _ALLOWED_MODES:
            raise ValueError(f"case {i} has invalid expected_mode: {exp}")
        out.append({"question": q, "expected_mode": exp})
    return out


def run_query_router_eval(
    *,
    cases_path: str,
    router_config_path: str | None,
    out_path: str | None = None,
    strict_min_accuracy: float | None = None,
) -> dict[str, Any]:
    if strict_min_accuracy is not None and not (0.0 <= float(strict_min_accuracy) <= 1.0):
        raise ValueError("strict_min_accuracy must be within [0.0, 1.0]")

    cfg = load_router_config(router_config_path)
    cases = _load_cases(cases_path)

    total = len(cases)
    correct = 0
    mismatches: list[dict[str, Any]] = []
    counts_expected: dict[str, int] = {m: 0 for m in sorted(_ALLOWED_MODES)}
    counts_predicted: dict[str, int] = {m: 0 for m in sorted(_ALLOWED_MODES)}
    confusion: dict[str, dict[str, int]] = {m: {x: 0 for x in sorted(_ALLOWED_MODES)} for m in sorted(_ALLOWED_MODES)}

    for i, row in enumerate(cases):
        question = row["question"]
        expected = row["expected_mode"]
        decision = route_query(question, cfg, explicit_mode=None)
        predicted = decision.mode

        counts_expected[expected] += 1
        if predicted not in counts_predicted:
            counts_predicted[predicted] = 0
        counts_predicted[predicted] += 1
        confusion.setdefault(expected, {})
        confusion[expected].setdefault(predicted, 0)
        confusion[expected][predicted] += 1

        if predicted == expected:
            correct += 1
        else:
            mismatches.append(
                {
                    "index": i,
                    "question": question,
                    "expected_mode": expected,
                    "predicted_mode": predicted,
                    "reason": decision.reason,
                    "signals": decision.signals,
                }
            )

    accuracy = (float(correct) / float(total)) if total else 0.0
    gate_applied = strict_min_accuracy is not None
    gate_passed = (accuracy >= float(strict_min_accuracy)) if gate_applied else True
    payload: dict[str, Any] = {
        "cases_path": str(Path(cases_path)),
        "router_config_path": str(router_config_path) if router_config_path else None,
        "total": total,
        "correct": correct,
        "accuracy": round(accuracy, 4),
        "strict_min_accuracy": strict_min_accuracy,
        "gate_applied": gate_applied,
        "gate_passed": gate_passed,
        "counts_expected": counts_expected,
        "counts_predicted": counts_predicted,
        "confusion": confusion,
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
    }

    if out_path:
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    return payload
