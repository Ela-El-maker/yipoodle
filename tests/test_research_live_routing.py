import json

from src.apps.index_builder import build_index
from src.apps.research_copilot import run_research
from src.core.live_schema import LiveItem


def test_auto_routing_chooses_finance_pack(monkeypatch, tmp_path) -> None:
    idx = tmp_path / "idx.json"
    build_index("tests/fixtures/extracted", str(idx))
    cfg = tmp_path / "sources.yaml"
    cfg.write_text(
        """
live_sources:
  coingecko_price:
    enabled: true
    type: rest
    endpoint: https://example.com/live
  hn_rss:
    enabled: true
    type: rss
    url: https://example.com/feed
live_routing:
  intents:
    finance_price:
      match_terms: [price, bitcoin]
      sources: [coingecko_price]
live_snapshot:
  root_dir: "{root}"
""".strip().format(root=str(tmp_path / "snap")),
        encoding="utf-8",
    )

    def fake_fetch(source, **_kwargs):
        if getattr(source, "name", "") == "coingecko_price":
            return ([LiveItem(id="x1", url="https://x", text="bitcoin price 50000 usd", source="coingecko_price")], "raw")
        return ([LiveItem(id="x2", url="https://y", text="generic headline", source="hn_rss")], "raw")

    monkeypatch.setattr("src.apps.research_copilot.fetch_live_source", fake_fetch)
    out = tmp_path / "r.md"
    run_research(str(idx), "What is bitcoin price now?", 6, str(out), min_items=1, min_score=0.0, retrieval_mode="lexical", sources_config_path=str(cfg), live_enabled=True)
    m = json.loads(out.with_suffix(".metrics.json").read_text(encoding="utf-8"))
    assert m["intent_detected"] == "finance_price"
    assert m["routed_sources"] == ["coingecko_price"]


def test_weather_without_pack_returns_not_found_diagnostics(tmp_path) -> None:
    idx = tmp_path / "idx.json"
    build_index("tests/fixtures/extracted", str(idx))
    cfg = tmp_path / "sources.yaml"
    cfg.write_text(
        """
live_sources:
  hn_rss:
    enabled: true
    type: rss
    url: https://example.com/feed
live_routing:
  intents:
    weather_now:
      match_terms: [weather]
      sources: [open_meteo]
live_routing_help:
  weather_now:
    suggest_enable: [open_meteo]
""".strip(),
        encoding="utf-8",
    )
    out = tmp_path / "w.md"
    run_research(str(idx), "What is the weather in Nairobi today?", 4, str(out), min_items=1, min_score=0.0, retrieval_mode="lexical", sources_config_path=str(cfg), live_enabled=True)
    txt = out.read_text(encoding="utf-8")
    assert "Not found in sources." in txt
    assert "Retrieval Diagnostics" in txt
    assert "suggest_enable" in txt
