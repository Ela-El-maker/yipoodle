from __future__ import annotations

from pathlib import Path
import json

from src.apps.pipeline_runner import run_full_pipeline


def test_run_full_pipeline_stops_on_source_gate(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        "src.apps.pipeline_runner.sync_papers",
        lambda **_kwargs: {"source_quality_healthy": False, "source_quality_reasons": ["x"]},
    )
    summary = run_full_pipeline(
        query="q",
        db_path=str(tmp_path / "db.sqlite"),
        papers_dir=str(tmp_path / "papers"),
        extracted_dir=str(tmp_path / "extracted"),
        index_path=str(tmp_path / "index.json"),
        report_path=str(tmp_path / "report.md"),
        sources_config_path=str(tmp_path / "sources.yaml"),
        fail_on_source_quality_gate=True,
    )
    assert summary["status"] == "failed"
    assert summary["failed_stage"] == "sync"


def test_run_full_pipeline_happy_path(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        "src.apps.pipeline_runner.sync_papers",
        lambda **_kwargs: {"source_quality_healthy": True, "source_quality_reasons": []},
    )
    monkeypatch.setattr(
        "src.apps.pipeline_runner.extract_from_papers_dir_with_db",
        lambda **_kwargs: {"processed": 1, "created": 1},
    )
    monkeypatch.setattr(
        "src.apps.pipeline_runner.evaluate_corpus_health",
        lambda **_kwargs: {"healthy": True, "reasons": []},
    )
    monkeypatch.setattr(
        "src.apps.pipeline_runner.build_index",
        lambda **_kwargs: {"snippets": 2},
    )

    report_path = tmp_path / "report.md"
    report_json_path = report_path.with_suffix(".json")
    evidence_path = report_path.with_suffix(".evidence.json")
    report_path.write_text("# Report\n", encoding="utf-8")
    report_json_path.write_text(
        json.dumps(
            {
                "question": "q",
                "shortlist": [{"paper_id": "p1", "title": "t", "reason": "r"}],
                "synthesis": "Claim (P1:S1)",
                "gaps": [],
                "experiments": [],
                "citations": ["(P1:S1)"],
            }
        ),
        encoding="utf-8",
    )
    evidence_path.write_text(
        json.dumps({"question": "q", "items": [{"paper_id": "p1", "snippet_id": "P1:S1", "score": 1.0, "section": "x", "text": "Claim"}]}),
        encoding="utf-8",
    )
    monkeypatch.setattr("src.apps.pipeline_runner.run_research", lambda **_kwargs: str(report_path))
    monkeypatch.setattr("src.apps.pipeline_runner.validate_report_citations", lambda _rep: [])
    monkeypatch.setattr("src.apps.pipeline_runner.validate_claim_support", lambda _rep, _ev: [])
    monkeypatch.setattr("src.apps.pipeline_runner.validate_no_new_numbers", lambda _txt, _ev: [])
    monkeypatch.setattr(
        "src.apps.pipeline_runner.report_coverage_metrics",
        lambda _rep, _ev: {"claim_lines": 1, "citation_coverage": 1.0},
    )

    summary = run_full_pipeline(
        query="q",
        db_path=str(tmp_path / "db.sqlite"),
        papers_dir=str(tmp_path / "papers"),
        extracted_dir=str(tmp_path / "extracted"),
        index_path=str(tmp_path / "index.json"),
        report_path=str(report_path),
        sources_config_path=str(tmp_path / "sources.yaml"),
        fail_on_source_quality_gate=True,
    )
    assert summary["status"] == "ok"
    assert summary["stages"]["validate_report"]["ok"] is True
