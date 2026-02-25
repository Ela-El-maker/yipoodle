from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import json
from typing import Any


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {}
    try:
        payload = json.loads(p.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _read_history(path: str | Path) -> list[dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return []
    try:
        payload = json.loads(p.read_text(encoding="utf-8"))
        return payload if isinstance(payload, list) else []
    except Exception:
        return []


def _extract_quality_metric(benchmark: dict[str, Any]) -> float | None:
    vec = (benchmark.get("vector_eval") or {}) if isinstance(benchmark, dict) else {}
    recall = vec.get("recall_at_k") if isinstance(vec, dict) else None
    try:
        if recall is None:
            return None
        return float(recall)
    except Exception:
        return None


def run_benchmark_regression_check(
    *,
    benchmark_path: str,
    history_path: str,
    run_id: str | None = None,
    max_latency_regression_pct: float = 10.0,
    min_quality_floor: float = 0.0,
    history_window: int = 104,
) -> dict[str, Any]:
    benchmark = _read_json(benchmark_path)
    if not benchmark:
        raise FileNotFoundError(f"benchmark file not found or invalid: {benchmark_path}")

    latency_no_cache = (benchmark.get("latency_no_cache") or {}) if isinstance(benchmark, dict) else {}
    curr_p95 = latency_no_cache.get("p95_ms") if isinstance(latency_no_cache, dict) else None
    try:
        curr_p95_val = float(curr_p95) if curr_p95 is not None else None
    except Exception:
        curr_p95_val = None

    curr_quality = _extract_quality_metric(benchmark)
    history = _read_history(history_path)
    previous = history[-1] if history else None
    prev_p95 = None
    if isinstance(previous, dict):
        try:
            prev_p95 = float(previous.get("p95_ms")) if previous.get("p95_ms") is not None else None
        except Exception:
            prev_p95 = None

    latency_regression_pct = None
    latency_regressed = False
    if curr_p95_val is not None and prev_p95 is not None and prev_p95 > 0:
        latency_regression_pct = ((curr_p95_val - prev_p95) / prev_p95) * 100.0
        latency_regressed = latency_regression_pct > float(max_latency_regression_pct)

    quality_below_floor = False
    if curr_quality is not None:
        quality_below_floor = curr_quality < float(min_quality_floor)

    regressed = bool(latency_regressed or quality_below_floor)
    reasons: list[str] = []
    if latency_regressed:
        reasons.append(
            f"latency_regression:{latency_regression_pct:.3f}%>{float(max_latency_regression_pct):.3f}%"
        )
    if quality_below_floor and curr_quality is not None:
        reasons.append(f"quality_below_floor:{curr_quality:.4f}<{float(min_quality_floor):.4f}")

    entry = {
        "run_id": run_id,
        "created_at": _utc_now(),
        "benchmark_path": str(Path(benchmark_path)),
        "retrieval_mode": benchmark.get("retrieval_mode"),
        "p95_ms": curr_p95_val,
        "quality_metric": curr_quality,
        "regressed": regressed,
        "reasons": reasons,
    }
    history.append(entry)
    if history_window > 0:
        history = history[-int(history_window):]
    out_history = Path(history_path)
    out_history.parent.mkdir(parents=True, exist_ok=True)
    out_history.write_text(json.dumps(history, indent=2), encoding="utf-8")

    return {
        "run_id": run_id,
        "benchmark_path": str(Path(benchmark_path)),
        "history_path": str(out_history),
        "history_count": len(history),
        "baseline_run_id": previous.get("run_id") if isinstance(previous, dict) else None,
        "baseline_p95_ms": prev_p95,
        "current_p95_ms": curr_p95_val,
        "latency_regression_pct": latency_regression_pct,
        "max_latency_regression_pct": float(max_latency_regression_pct),
        "quality_metric": curr_quality,
        "min_quality_floor": float(min_quality_floor),
        "quality_below_floor": quality_below_floor,
        "regressed": regressed,
        "reasons": reasons,
    }
