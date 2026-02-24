from pathlib import Path

from src.apps.index_builder import build_index
from src.apps.research_copilot import run_research


def test_build_index_then_research_end_to_end(tmp_path) -> None:
    index_path = tmp_path / "bm25.json"
    stats = build_index("tests/fixtures/extracted", str(index_path))
    assert "enriched" in stats
    out = tmp_path / "report.md"
    run_research(str(index_path), "What are limitations in mobile segmentation?", 4, str(out))

    assert out.exists()
    evidence_path = out.with_suffix(".evidence.json")
    assert evidence_path.exists()
    assert "(P" in out.read_text(encoding="utf-8")
