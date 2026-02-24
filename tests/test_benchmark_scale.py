import json
from pathlib import Path

from src.apps.benchmark import benchmark_scale


def test_benchmark_scale_outputs_payload(tmp_path) -> None:
    out = tmp_path / "bench_scale.json"
    p = benchmark_scale(
        corpus_dir="tests/fixtures/extracted",
        queries=["mobile segmentation", "limitations"],
        repeat_factor=5,
        runs_per_query=1,
        out_path=str(out),
    )
    data = json.loads(Path(p).read_text(encoding="utf-8"))
    assert data["expanded_snippets"] >= data["base_snippets"]
    assert "latency_ms" in data
