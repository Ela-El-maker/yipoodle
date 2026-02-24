import json

from src.apps.index_builder import build_index
from src.apps.research_copilot import run_research


def test_manual_mode_without_live_sources_uses_none(tmp_path) -> None:
    idx = tmp_path / "idx.json"
    build_index("tests/fixtures/extracted", str(idx))
    cfg = tmp_path / "sources.yaml"
    cfg.write_text(
        """
live_sources:
  demo_live:
    enabled: true
    type: rest
    endpoint: https://example.com/live
""".strip(),
        encoding="utf-8",
    )
    out = tmp_path / "manual.md"
    run_research(
        str(idx),
        "What is weather in Nairobi?",
        4,
        str(out),
        min_items=1,
        min_score=0.0,
        retrieval_mode="lexical",
        sources_config_path=str(cfg),
        live_enabled=True,
        routing_mode="manual",
    )
    m = json.loads(out.with_suffix(".metrics.json").read_text(encoding="utf-8"))
    assert m["routing_mode"] == "manual"
    assert m["routed_sources"] == []
