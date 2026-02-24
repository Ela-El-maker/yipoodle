from __future__ import annotations

from pathlib import Path
import json
from typing import Any

from src.apps.corpus_health import evaluate_corpus_health
from src.apps.evidence_extract import extract_from_papers_dir_with_db
from src.apps.extraction_eval import write_extraction_eval_report
from src.apps.index_builder import build_index
from src.apps.paper_sync import sync_papers
from src.apps.research_copilot import run_research
from src.core.schemas import EvidencePack, ResearchReport
from src.core.validation import (
    report_coverage_metrics,
    validate_claim_support,
    validate_no_new_numbers,
    validate_report_citations,
)


def _utc_now_iso() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def run_full_pipeline(
    *,
    query: str,
    db_path: str,
    papers_dir: str,
    extracted_dir: str,
    index_path: str,
    report_path: str,
    sources_config_path: str,
    max_results: int = 20,
    with_semantic_scholar: bool = True,
    top_k: int = 8,
    min_items: int = 2,
    min_score: float = 0.5,
    min_text_chars: int = 200,
    min_snippets: int = 1,
    min_avg_chars_per_paper: int = 500,
    min_avg_chars_per_page: int = 80,
    max_extract_error_rate: float = 0.8,
    require_healthy_corpus: bool = True,
    fail_on_source_quality_gate: bool = True,
    extraction_gold_path: str | None = None,
    extraction_eval_fail_below: float | None = None,
) -> dict[str, Any]:
    summary: dict[str, Any] = {"query": query, "status": "ok", "created_utc": _utc_now_iso(), "run_id": "pipeline_manual", "stages": {}}

    sync_stats = sync_papers(
        query=query,
        max_results=max_results,
        db_path=db_path,
        papers_dir=papers_dir,
        with_semantic_scholar=with_semantic_scholar,
        sources_config_path=sources_config_path,
    )
    summary["stages"]["sync"] = sync_stats
    if fail_on_source_quality_gate and not bool(sync_stats.get("source_quality_healthy", True)):
        summary["status"] = "failed"
        summary["failed_stage"] = "sync"
        summary["reason"] = "source_quality_gate_failed"
        return summary

    extract_stats = extract_from_papers_dir_with_db(
        papers_dir=papers_dir,
        out_dir=extracted_dir,
        db_path=db_path,
        min_text_chars=min_text_chars,
    )
    summary["stages"]["extract"] = extract_stats

    health = evaluate_corpus_health(
        corpus_dir=extracted_dir,
        extract_stats_path=None,
        min_snippets=min_snippets,
        min_avg_chars_per_paper=min_avg_chars_per_paper,
        min_avg_chars_per_page=min_avg_chars_per_page,
        max_extract_error_rate=max_extract_error_rate,
    )
    summary["stages"]["corpus_health"] = health
    if require_healthy_corpus and not bool(health.get("healthy", False)):
        summary["status"] = "failed"
        summary["failed_stage"] = "corpus_health"
        summary["reason"] = "corpus_health_gate_failed"
        return summary

    index_stats = build_index(
        corpus_dir=extracted_dir,
        out_path=index_path,
        db_path=db_path,
        require_healthy_corpus=require_healthy_corpus,
        min_snippets=min_snippets,
        min_avg_chars_per_paper=min_avg_chars_per_paper,
        min_avg_chars_per_page=min_avg_chars_per_page,
        max_extract_error_rate=max_extract_error_rate,
    )
    summary["stages"]["build_index"] = index_stats

    report_out = run_research(
        index_path=index_path,
        question=query,
        top_k=top_k,
        out_path=report_path,
        min_items=min_items,
        min_score=min_score,
        sources_config_path=sources_config_path,
    )
    summary["stages"]["research"] = {"report_path": report_out}

    evidence_path = str(Path(report_path).with_suffix(".evidence.json"))
    report_json_path = str(Path(report_path).with_suffix(".json"))
    errors: list[str] = []
    metrics: dict[str, Any] = {}
    if Path(report_json_path).exists() and Path(evidence_path).exists():
        rep = ResearchReport(**json.loads(Path(report_json_path).read_text(encoding="utf-8")))
        evidence = EvidencePack(**json.loads(Path(evidence_path).read_text(encoding="utf-8")))
        errors.extend(validate_report_citations(rep))
        errors.extend(validate_claim_support(rep, evidence))
        text = rep.synthesis + "\n" + "\n".join(rep.gaps + [e.proposal for e in rep.experiments])
        errors.extend(validate_no_new_numbers(text, evidence))
        metrics = report_coverage_metrics(rep, evidence)
    summary["stages"]["validate_report"] = {"ok": len(errors) == 0, "errors": errors, "metrics": metrics}
    if errors:
        summary["status"] = "failed"
        summary["failed_stage"] = "validate_report"
        summary["reason"] = "report_validation_failed"
        return summary

    if extraction_gold_path:
        eval_out = str(Path(report_path).with_suffix(".extraction_eval.md"))
        _out_path, eval_report = write_extraction_eval_report(
            corpus_dir=extracted_dir,
            gold_path=extraction_gold_path,
            out_path=eval_out,
        )
        score = float((eval_report.get("summary") or {}).get("weighted_score", 0.0))
        summary["stages"]["extraction_eval"] = {
            "report_path": _out_path,
            "summary": eval_report.get("summary", {}),
        }
        if extraction_eval_fail_below is not None and score < float(extraction_eval_fail_below):
            summary["status"] = "failed"
            summary["failed_stage"] = "extraction_eval"
            summary["reason"] = (
                f"extraction_eval_score_below_threshold:{score:.4f}<{float(extraction_eval_fail_below):.4f}"
            )
            return summary

    return summary
