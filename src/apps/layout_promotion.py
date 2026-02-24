from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import tempfile

from src.apps.evidence_extract import extract_from_papers_dir, extract_from_papers_dir_with_db
from src.apps.extraction_eval import evaluate_extraction_against_gold


@dataclass
class _EvalRates:
    weighted_score: float
    ordered_pass_rate: float
    page_nonempty_pass_rate: float


def _check_pass_rate(report: dict[str, object], check_type: str) -> float:
    rows = report.get("results", []) or []
    checks = [c for r in rows for c in (r.get("checks", []) or []) if str((c or {}).get("type", "")) == check_type]
    if not checks:
        return 1.0
    passed = sum(1 for c in checks if bool(c.get("passed", False)))
    return passed / float(len(checks))


def _eval_rates(report: dict[str, object]) -> _EvalRates:
    summary = report.get("summary", {}) or {}
    return _EvalRates(
        weighted_score=float(summary.get("weighted_score", 0.0) or 0.0),
        ordered_pass_rate=_check_pass_rate(report, "ordered_contains"),
        page_nonempty_pass_rate=_check_pass_rate(report, "page_nonempty_ratio"),
    )


def run_layout_promotion_gate(
    *,
    papers_dir: str,
    gold_path: str,
    state_path: str = "runs/audit/layout_promotion_state.json",
    db_path: str | None = None,
    min_text_chars: int = 200,
    two_column_mode: str = "auto",
    min_weighted_score: float = 0.75,
    max_weighted_regression: float = 0.02,
    max_ordered_regression: float = 0.02,
    max_page_nonempty_regression: float = 0.02,
) -> dict[str, object]:
    with tempfile.TemporaryDirectory() as td:
        legacy_dir = Path(td) / "legacy"
        v2_dir = Path(td) / "v2"
        legacy_dir.mkdir(parents=True, exist_ok=True)
        v2_dir.mkdir(parents=True, exist_ok=True)

        kwargs = {
            "papers_dir": papers_dir,
            "min_text_chars": min_text_chars,
            "two_column_mode": two_column_mode,
            "ocr_enabled": False,
        }
        if db_path:
            legacy_stats = extract_from_papers_dir_with_db(out_dir=str(legacy_dir), db_path=db_path, layout_engine="legacy", **kwargs)
            v2_stats = extract_from_papers_dir_with_db(out_dir=str(v2_dir), db_path=db_path, layout_engine="v2", **kwargs)
        else:
            legacy_stats = extract_from_papers_dir(out_dir=str(legacy_dir), layout_engine="legacy", **kwargs)
            v2_stats = extract_from_papers_dir(out_dir=str(v2_dir), layout_engine="v2", **kwargs)

        legacy_eval = evaluate_extraction_against_gold(str(legacy_dir), gold_path)
        v2_eval = evaluate_extraction_against_gold(str(v2_dir), gold_path)

    l = _eval_rates(legacy_eval)
    v = _eval_rates(v2_eval)
    checks = {
        "min_weighted_score": v.weighted_score >= float(min_weighted_score),
        "weighted_non_regression": v.weighted_score + float(max_weighted_regression) >= l.weighted_score,
        "ordered_non_regression": v.ordered_pass_rate + float(max_ordered_regression) >= l.ordered_pass_rate,
        "page_nonempty_non_regression": v.page_nonempty_pass_rate + float(max_page_nonempty_regression)
        >= l.page_nonempty_pass_rate,
    }
    promoted = all(bool(x) for x in checks.values())
    out = {
        "promoted": promoted,
        "recommended_layout_engine": "v2" if promoted else "shadow",
        "checks": checks,
        "thresholds": {
            "min_weighted_score": float(min_weighted_score),
            "max_weighted_regression": float(max_weighted_regression),
            "max_ordered_regression": float(max_ordered_regression),
            "max_page_nonempty_regression": float(max_page_nonempty_regression),
        },
        "legacy": {
            "weighted_score": round(l.weighted_score, 4),
            "ordered_pass_rate": round(l.ordered_pass_rate, 4),
            "page_nonempty_pass_rate": round(l.page_nonempty_pass_rate, 4),
            "extract_stats": legacy_stats,
        },
        "v2": {
            "weighted_score": round(v.weighted_score, 4),
            "ordered_pass_rate": round(v.ordered_pass_rate, 4),
            "page_nonempty_pass_rate": round(v.page_nonempty_pass_rate, 4),
            "extract_stats": v2_stats,
        },
    }
    p = Path(state_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out
