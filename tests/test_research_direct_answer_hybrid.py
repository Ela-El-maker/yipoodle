import json

from src.apps.index_builder import build_index
from src.apps.research_copilot import run_research


def test_research_direct_answer_hybrid_math(tmp_path) -> None:
    idx = tmp_path / "idx.json"
    build_index("tests/fixtures/extracted", str(idx))
    out = tmp_path / "math.md"
    run_research(
        str(idx),
        "23 + 34 = ?",
        4,
        str(out),
        min_items=1,
        min_score=0.0,
        retrieval_mode="lexical",
        live_enabled=True,
        direct_answer_mode="hybrid",
    )
    txt = out.read_text(encoding="utf-8")
    metrics = json.loads(out.with_suffix(".metrics.json").read_text(encoding="utf-8"))
    assert "Direct answer: 57" in txt
    assert metrics["direct_answer_used"] is True
    assert metrics["direct_answer_value"] == "57"
