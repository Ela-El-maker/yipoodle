from __future__ import annotations

from pathlib import Path
import json

from src.apps.benchmark_regression import run_benchmark_regression_check


def _write_benchmark(path: Path, p95_ms: float, recall: float | None = 1.0) -> None:
    payload = {
        "retrieval_mode": "hybrid",
        "latency_no_cache": {"p95_ms": p95_ms},
        "vector_eval": {"recall_at_k": recall},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def test_benchmark_regression_first_run_has_no_baseline(tmp_path) -> None:
    bench = tmp_path / "bench.json"
    hist = tmp_path / "history.json"
    _write_benchmark(bench, 100.0, 0.99)
    out = run_benchmark_regression_check(
        benchmark_path=str(bench),
        history_path=str(hist),
        run_id="r1",
        max_latency_regression_pct=10.0,
        min_quality_floor=0.5,
    )
    assert out["regressed"] is False
    assert out["baseline_run_id"] is None
    assert hist.exists()


def test_benchmark_regression_detects_latency_drift(tmp_path) -> None:
    bench = tmp_path / "bench.json"
    hist = tmp_path / "history.json"
    _write_benchmark(bench, 100.0, 0.99)
    _ = run_benchmark_regression_check(
        benchmark_path=str(bench),
        history_path=str(hist),
        run_id="r1",
        max_latency_regression_pct=10.0,
        min_quality_floor=0.5,
    )
    _write_benchmark(bench, 130.0, 0.99)
    out = run_benchmark_regression_check(
        benchmark_path=str(bench),
        history_path=str(hist),
        run_id="r2",
        max_latency_regression_pct=10.0,
        min_quality_floor=0.5,
    )
    assert out["regressed"] is True
    assert "latency_regression" in " ".join(out["reasons"])


def test_benchmark_regression_detects_quality_floor(tmp_path) -> None:
    bench = tmp_path / "bench.json"
    hist = tmp_path / "history.json"
    _write_benchmark(bench, 100.0, 0.3)
    out = run_benchmark_regression_check(
        benchmark_path=str(bench),
        history_path=str(hist),
        run_id="r1",
        max_latency_regression_pct=99.0,
        min_quality_floor=0.5,
    )
    assert out["regressed"] is True
    assert out["quality_below_floor"] is True
