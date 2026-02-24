import json

from src.apps.index_builder import build_index
from src.apps.research_copilot import run_research
from src.core.live_schema import LiveItem


def test_research_live_union_merges_and_writes_metrics(monkeypatch, tmp_path) -> None:
    idx_path = tmp_path / "idx.json"
    build_index("tests/fixtures/extracted", str(idx_path))

    cfg = tmp_path / "sources.yaml"
    cfg.write_text(
        """
live_sources:
  demo_live:
    enabled: true
    type: rest
    endpoint: https://example.com/live
    domain_tags: [computer_vision]
live_snapshot:
  root_dir: "{root}"
""".strip(),
        encoding="utf-8",
    )
    cfg.write_text(cfg.read_text(encoding="utf-8").format(root=str(tmp_path / "snapshots")), encoding="utf-8")

    def fake_fetch_live_source(*_args, **_kwargs):
        return ([LiveItem(id="x1", url="https://x", text="fresh live evidence", source="demo_live")], "raw")

    monkeypatch.setattr("src.apps.research_copilot.fetch_live_source", fake_fetch_live_source)

    out = tmp_path / "report_live.md"
    run_research(
        str(idx_path),
        "mobile segmentation limitations",
        5,
        str(out),
        min_items=1,
        min_score=0.0,
        retrieval_mode="lexical",
        sources_config_path=str(cfg),
        live_enabled=True,
        live_merge_mode="union",
    )
    metrics = json.loads(out.with_suffix(".metrics.json").read_text(encoding="utf-8"))
    evidence = json.loads(out.with_suffix(".evidence.json").read_text(encoding="utf-8"))
    assert metrics["live_enabled"] is True
    assert metrics["live_fetch_succeeded"] >= 1
    assert metrics["live_snippet_count"] >= 1
    assert any(str(x).startswith("SNAP:") for x in [i["snippet_id"] for i in evidence["items"]])
