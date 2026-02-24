import json
from pathlib import Path

from src.apps.benchmark import benchmark_research
from src.apps.index_builder import build_index
from src.apps.research_copilot import run_research
from src.apps.retrieval import clear_index_cache, index_cache_info, load_index


def test_index_cache_hits(tmp_path) -> None:
    idx_path = tmp_path / "index.json"
    build_index("tests/fixtures/extracted", str(idx_path))

    clear_index_cache()
    _ = load_index(str(idx_path))
    _ = load_index(str(idx_path))
    info = index_cache_info()
    assert info["hits"] >= 1


def test_research_query_cache_hit_metric(tmp_path) -> None:
    idx_path = tmp_path / "index.json"
    build_index("tests/fixtures/extracted", str(idx_path))

    out1 = tmp_path / "r1.md"
    out2 = tmp_path / "r2.md"
    cache_dir = tmp_path / "cache"

    run_research(str(idx_path), "mobile segmentation limitations", 4, str(out1), min_items=1, min_score=0.1, use_cache=True, cache_dir=str(cache_dir))
    run_research(str(idx_path), "mobile segmentation limitations", 4, str(out2), min_items=1, min_score=0.1, use_cache=True, cache_dir=str(cache_dir))

    m2 = json.loads(out2.with_suffix(".metrics.json").read_text(encoding="utf-8"))
    assert m2["cache_hit"] is True


def test_benchmark_output(tmp_path) -> None:
    idx_path = tmp_path / "index.json"
    build_index("tests/fixtures/extracted", str(idx_path))

    out = tmp_path / "bench.json"
    p = benchmark_research(
        index_path=str(idx_path),
        queries=["mobile segmentation", "boundary limitations"],
        runs_per_query=1,
        out_path=str(out),
        min_items=1,
        min_score=0.1,
    )
    data = json.loads(Path(p).read_text(encoding="utf-8"))
    assert data["num_queries"] == 2
    assert "latency_no_cache" in data
    assert "latency_with_cache" in data
