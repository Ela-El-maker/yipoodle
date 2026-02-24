import json

from src.apps.index_builder import build_index
from src.apps.research_copilot import _apply_quality_to_vector_scores, run_research
from src.core.schemas import SnippetRecord


def test_research_vector_mode_with_stubbed_vector_backend(monkeypatch, tmp_path) -> None:
    idx_path = tmp_path / "idx.json"
    build_index("tests/fixtures/extracted", str(idx_path))

    class _Loaded:
        def __init__(self, snippet_ids):
            self.snippet_ids = snippet_ids

    def fake_load_vector_index(_idx, _meta):
        payload = json.loads(idx_path.read_text(encoding="utf-8"))
        return _Loaded([s["snippet_id"] for s in payload["snippets"]])

    def fake_query_vector_index(_loaded, question, top_k, model_name_override=None):
        del question, model_name_override
        payload = json.loads(idx_path.read_text(encoding="utf-8"))
        return [(s["snippet_id"], 1.0 - i * 0.01) for i, s in enumerate(payload["snippets"][:top_k])]

    monkeypatch.setattr("src.apps.research_copilot.load_vector_index", fake_load_vector_index)
    monkeypatch.setattr("src.apps.research_copilot.query_vector_index", fake_query_vector_index)

    out = tmp_path / "report_vector.md"
    run_research(
        str(idx_path),
        "mobile segmentation limitations",
        4,
        str(out),
        min_items=1,
        min_score=0.0,
        retrieval_mode="vector",
    )
    metrics = json.loads(out.with_suffix(".metrics.json").read_text(encoding="utf-8"))
    assert metrics["retrieval_mode"] == "vector"
    assert metrics["vector_index_loaded"] is True


def test_research_hybrid_fallback_on_vector_mismatch(monkeypatch, tmp_path) -> None:
    idx_path = tmp_path / "idx.json"
    build_index("tests/fixtures/extracted", str(idx_path))

    class _Loaded:
        snippet_ids = ["mismatch_only"]

    monkeypatch.setattr("src.apps.research_copilot.load_vector_index", lambda *_args, **_kwargs: _Loaded())

    out = tmp_path / "report_hybrid.md"
    run_research(
        str(idx_path),
        "mobile segmentation limitations",
        4,
        str(out),
        min_items=1,
        min_score=0.0,
        retrieval_mode="hybrid",
    )
    metrics = json.loads(out.with_suffix(".metrics.json").read_text(encoding="utf-8"))
    assert metrics["retrieval_mode"] == "hybrid"
    assert metrics["vector_mismatch"] is True


def test_vector_mode_applies_extraction_quality_prior(monkeypatch, tmp_path) -> None:
    s1 = SnippetRecord(
        snippet_id="Pp1:S1",
        paper_id="p1",
        section="results",
        text="a",
        token_count=1,
        extraction_quality_score=0.0,
    )
    s2 = SnippetRecord(
        snippet_id="Pp2:S1",
        paper_id="p2",
        section="results",
        text="b",
        token_count=1,
        extraction_quality_score=1.0,
    )
    adjusted = _apply_quality_to_vector_scores(
        {"Pp1:S1": 1.0, "Pp2:S1": 0.95},
        {"Pp1:S1": s1, "Pp2:S1": s2},
        quality_prior_weight=0.15,
    )
    assert adjusted["Pp2:S1"] > adjusted["Pp1:S1"]


def test_research_applies_sources_yaml_limits_and_ranking(tmp_path) -> None:
    idx_path = tmp_path / "idx.json"
    build_index("tests/fixtures/extracted", str(idx_path))
    src_cfg = tmp_path / "sources.yaml"
    src_cfg.write_text(
        """
ranking:
  weights:
    recency: 0.40
    citation_count: 0.30
    source_trust: 0.20
limits:
  max_tokens_per_summary: 5
""".strip(),
        encoding="utf-8",
    )

    out = tmp_path / "report_sources.md"
    run_research(
        str(idx_path),
        "mobile segmentation limitations",
        4,
        str(out),
        min_items=1,
        min_score=0.0,
        retrieval_mode="lexical",
        sources_config_path=str(src_cfg),
    )
    metrics = json.loads(out.with_suffix(".metrics.json").read_text(encoding="utf-8"))
    report = json.loads(out.with_suffix(".json").read_text(encoding="utf-8"))
    assert metrics["sources_config_applied"] == 1
    assert metrics["max_tokens_per_summary"] == 5
    assert metrics["metadata_prior_weight"] == 0.45
    assert len(str(report["synthesis"]).split()) <= 5


def test_research_writes_semantic_metrics(monkeypatch, tmp_path) -> None:
    idx_path = tmp_path / "idx.json"
    build_index("tests/fixtures/extracted", str(idx_path))
    monkeypatch.setattr(
        "src.apps.research_copilot.validate_semantic_claim_support",
        lambda *_args, **_kwargs: (
            [],
            {
                "semantic_checked": True,
                "semantic_model": "sentence-transformers/all-MiniLM-L6-v2",
                "semantic_support_avg": 0.77,
                "semantic_support_min": 0.61,
                "semantic_contradiction_max": 0.12,
                "semantic_lines_below_threshold": 0,
                "semantic_status": "pass",
                "semantic_shadow_mode": True,
            },
            [],
        ),
    )
    out = tmp_path / "report_sem.md"
    run_research(
        str(idx_path),
        "mobile segmentation limitations",
        4,
        str(out),
        min_items=1,
        min_score=0.0,
        retrieval_mode="lexical",
    )
    metrics = json.loads(out.with_suffix(".metrics.json").read_text(encoding="utf-8"))
    assert metrics["semantic_checked"] is True
    assert metrics["semantic_status"] == "pass"


def test_research_relevance_gate_blocks_irrelevant_live_evidence(monkeypatch, tmp_path) -> None:
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
live_snapshot:
  root_dir: "{root}"
""".strip().format(root=str(tmp_path / "snapshots")),
        encoding="utf-8",
    )

    from src.core.live_schema import LiveItem

    def fake_fetch_live_source(*_args, **_kwargs):
        return ([LiveItem(id="x1", url="https://x", text="GPU startup benchmark notes", source="demo_live")], "raw")

    monkeypatch.setattr("src.apps.research_copilot.fetch_live_source", fake_fetch_live_source)

    out = tmp_path / "report_irrel.md"
    run_research(
        str(idx_path),
        "What is the weather in Nairobi today?",
        4,
        str(out),
        min_items=1,
        min_score=0.0,
        retrieval_mode="lexical",
        sources_config_path=str(cfg),
        live_enabled=True,
        live_sources_override="demo_live",
    )
    report = out.read_text(encoding="utf-8")
    metrics = json.loads(out.with_suffix(".metrics.json").read_text(encoding="utf-8"))
    assert "Not found in sources." in report
    assert metrics["question_relevance_gate_passed"] is False


def test_report_key_claims_include_citations_for_kb_ingest(tmp_path) -> None:
    idx_path = tmp_path / "idx.json"
    build_index("tests/fixtures/extracted", str(idx_path))
    out = tmp_path / "report_key_claims.md"
    run_research(
        str(idx_path),
        "mobile segmentation limitations",
        4,
        str(out),
        min_items=1,
        min_score=0.0,
        retrieval_mode="lexical",
    )
    md = out.read_text(encoding="utf-8")
    assert "## Key Claims" in md
    key_claim_block = md.split("## Key Claims", 1)[1].split("## Gaps", 1)[0]
    assert ":S" in key_claim_block
