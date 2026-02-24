import json

from src.apps.index_builder import build_index
from src.apps.research_copilot import run_research
from src.core.live_schema import LiveItem


def test_reject_reason_no_sources_for_intent(tmp_path) -> None:
    idx = tmp_path / "idx.json"
    build_index("tests/fixtures/extracted", str(idx))
    cfg = tmp_path / "sources.yaml"
    cfg.write_text(
        """
live_sources:
  hn_rss:
    enabled: true
    type: rss
    url: https://example.com
live_routing:
  intents:
    weather_now:
      match_terms: [weather]
      sources: [open_meteo]
""".strip(),
        encoding="utf-8",
    )
    out = tmp_path / "r.md"
    run_research(str(idx), "what is the weather", 4, str(out), min_items=1, min_score=0.0, retrieval_mode="lexical", sources_config_path=str(cfg), live_enabled=True)
    m = json.loads(out.with_suffix(".metrics.json").read_text(encoding="utf-8"))
    assert m["relevance_reject_reason"] == "no_sources_for_intent"


def test_reject_reason_low_relevance(monkeypatch, tmp_path) -> None:
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
live_routing:
  intents:
    weather_now:
      match_terms: [weather]
      sources: [demo_live]
live_snapshot:
  root_dir: "{root}"
""".strip().format(root=str(tmp_path / "snap")),
        encoding="utf-8",
    )

    def fake_fetch_live_source(*_args, **_kwargs):
        return ([LiveItem(id="x1", url="https://x", text="gpu startup benchmark", source="demo_live")], "raw")

    monkeypatch.setattr("src.apps.research_copilot.fetch_live_source", fake_fetch_live_source)
    out = tmp_path / "r2.md"
    run_research(str(idx), "weather in nairobi", 4, str(out), min_items=1, min_score=0.0, retrieval_mode="lexical", sources_config_path=str(cfg), live_enabled=True)
    m = json.loads(out.with_suffix(".metrics.json").read_text(encoding="utf-8"))
    assert m["relevance_reject_reason"] == "low_relevance"


def test_reject_reason_empty_live_results(monkeypatch, tmp_path) -> None:
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
live_routing:
  intents:
    finance_price:
      match_terms: [price]
      sources: [demo_live]
live_snapshot:
  root_dir: "{root}"
""".strip().format(root=str(tmp_path / "snap")),
        encoding="utf-8",
    )

    def fake_fetch_live_source(*_args, **_kwargs):
        return ([], "raw")

    monkeypatch.setattr("src.apps.research_copilot.fetch_live_source", fake_fetch_live_source)
    out = tmp_path / "r3.md"
    run_research(str(idx), "bitcoin price", 4, str(out), min_items=1, min_score=0.0, retrieval_mode="lexical", sources_config_path=str(cfg), live_enabled=True)
    m = json.loads(out.with_suffix(".metrics.json").read_text(encoding="utf-8"))
    assert m["relevance_reject_reason"] == "empty_live_results"
