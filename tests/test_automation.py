from pathlib import Path
import json
import sqlite3

from src.apps.automation import load_automation_config, parse_run_summary, write_summary_outputs


def test_load_automation_config_defaults(tmp_path) -> None:
    cfg_path = tmp_path / "automation.yaml"
    cfg_path.write_text(
        """
topics:
  - name: topic_a
    query: q1
""".strip(),
        encoding="utf-8",
    )
    cfg = load_automation_config(str(cfg_path))
    assert cfg["global"]["prefer_arxiv"] is True
    assert cfg["global"]["layout_engine"] == "shadow"
    assert cfg["global"]["ocr_enabled"] is None
    assert cfg["global"]["ocr_lang"] is None
    assert cfg["thresholds"]["min_citation_coverage"] == 1.0
    assert cfg["thresholds"]["semantic_enabled"] is True
    assert cfg["thresholds"]["semantic_shadow_mode"] is True
    assert len(cfg["topics"]) == 1


def test_parse_and_write_summary(tmp_path) -> None:
    audit_dir = tmp_path / "runs" / "audit"
    run_dir = audit_dir / "runs" / "20260220T020000Z"
    run_dir.mkdir(parents=True)

    db_path = tmp_path / "data" / "papers.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.execute("CREATE TABLE papers (paper_id TEXT PRIMARY KEY)")
        conn.execute("INSERT INTO papers(paper_id) VALUES ('p1')")
        conn.execute("INSERT INTO papers(paper_id) VALUES ('p2')")
        conn.commit()

    papers_dir = tmp_path / "data" / "papers"
    papers_dir.mkdir(parents=True, exist_ok=True)
    (papers_dir / "p1.pdf").write_bytes(b"%PDF-1.4")

    index_path = tmp_path / "data" / "indexes" / "idx.json"
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text(json.dumps({"snippets": [{"snippet_id": "Pp1:S1"}]}), encoding="utf-8")

    sync_stats = {
        "fetched": 10,
        "added": 5,
        "downloaded": 2,
        "source_errors": 0,
        "missing_pdf_url": 1,
        "download_http_error": 2,
        "non_pdf_content_type": 1,
        "blocked_or_paywalled": 0,
        "require_pdf_filtered": 0,
    }
    sync_path = run_dir / "sync_topic.stats.json"
    sync_path.write_text(json.dumps(sync_stats), encoding="utf-8")

    extract_path = run_dir / "extract.stats.json"
    extract_path.write_text(
        json.dumps(
            {
                "processed": 2,
                "created": 1,
                "resolved_from_db": 1,
                "failed_pdfs_count": 1,
                "failed_reason_counts": {"low_text": 1},
                "extractor_used_counts": {"pypdf": 1},
            }
        ),
        encoding="utf-8",
    )
    index_stats_path = run_dir / "index.stats.json"
    index_stats_path.write_text(json.dumps({"snippets": 1, "enriched": 1}), encoding="utf-8")

    metrics_path = tmp_path / "runs" / "research_reports" / "topic.metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(
        json.dumps(
            {
                "citation_coverage": 1.0,
                "evidence_usage": 0.75,
                "semantic_status": "warn",
                "semantic_support_avg": 0.51,
                "semantic_support_min": 0.42,
                "semantic_contradiction_max": 0.44,
                "semantic_lines_below_threshold": 2,
            }
        ),
        encoding="utf-8",
    )

    manifest = {
        "run_id": "20260220T020000Z",
        "created_utc": "2026-02-20T02:00:00Z",
        "paths": {
            "db_path": str(db_path),
            "papers_dir": str(papers_dir),
            "index_path": str(index_path),
        },
        "topics": [
            {
                "name": "topic",
                "query": "q",
                "sync_stats_file": str(sync_path),
                "validate_ok": True,
                "metrics_path": str(metrics_path),
            }
        ],
        "extract_stats_file": str(extract_path),
        "index_stats_file": str(index_stats_path),
        "benchmark_file": "",
        "snapshot_path": "",
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    summary = parse_run_summary(run_dir)
    assert summary["sync"]["pdf_download_rate"] == 0.2
    assert summary["corpus_profile"]["total_papers_in_db"] == 2
    assert summary["corpus_profile"]["total_downloaded_pdfs"] == 1
    assert summary["extract"]["extract_success_rate"] == 0.5
    assert summary["extract"]["failed_reason_counts"]["low_text"] == 1
    assert summary["topics"][0]["validate_ok"] is True
    assert summary["topics"][0]["semantic_status"] == "warn"
    assert summary["topics"][0]["semantic_lines_below_threshold"] == 2

    out_json = audit_dir / "latest_summary.json"
    out_md = audit_dir / "latest_summary.md"
    written = write_summary_outputs(audit_dir=str(audit_dir), run_dir=str(run_dir), out_json=str(out_json), out_md=str(out_md))
    assert written["run_id"] == "20260220T020000Z"
    assert out_json.exists()
    assert out_md.exists()
    md = out_md.read_text(encoding="utf-8")
    assert "## Semantic Faithfulness" in md
    assert "semantic_status=warn" in md
