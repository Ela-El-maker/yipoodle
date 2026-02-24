from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from src.apps.automation import parse_run_summary
from src.apps.monitoring_engine import monitor_digest_flush, monitor_status


def test_parse_summary_includes_monitoring(tmp_path) -> None:
    audit_dir = tmp_path / "runs" / "audit"
    run_dir = audit_dir / "runs" / "20260224T120000Z"
    run_dir.mkdir(parents=True)

    db_path = tmp_path / "data" / "papers.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.execute("CREATE TABLE papers (paper_id TEXT PRIMARY KEY)")
        conn.execute("INSERT INTO papers(paper_id) VALUES ('p1')")
        conn.commit()

    papers_dir = tmp_path / "data" / "papers"
    papers_dir.mkdir(parents=True)
    (papers_dir / "p1.pdf").write_bytes(b"%PDF-1.4")

    index_path = tmp_path / "idx.json"
    index_path.write_text(json.dumps({"snippets": []}), encoding="utf-8")
    sync_path = run_dir / "sync.stats.json"
    sync_path.write_text(json.dumps({"fetched": 0, "added": 0, "downloaded": 0, "source_errors": 0}), encoding="utf-8")
    extract_path = run_dir / "extract.stats.json"
    extract_path.write_text(json.dumps({"processed": 0, "created": 0}), encoding="utf-8")
    index_stats_path = run_dir / "index.stats.json"
    index_stats_path.write_text(json.dumps({"snippets": 0}), encoding="utf-8")
    metrics_path = tmp_path / "topic.metrics.json"
    metrics_path.write_text(json.dumps({"citation_coverage": 1.0, "evidence_usage": 1.0}), encoding="utf-8")

    manifest = {
        "run_id": "20260224T120000Z",
        "created_utc": "2026-02-24T12:00:00Z",
        "paths": {"db_path": str(db_path), "papers_dir": str(papers_dir), "index_path": str(index_path)},
        "topics": [
            {
                "name": "finance",
                "query": "q",
                "sync_stats_file": str(sync_path),
                "validate_ok": True,
                "metrics_path": str(metrics_path),
                "monitoring_enabled": True,
                "monitor_status": "warn",
                "monitor_events_count": 1,
                "monitor_errors_count": 0,
                "monitor_events": [
                    {"severity": "critical", "code": "monitor_trigger_high", "message": "x", "details": {}}
                ],
            }
        ],
        "extract_stats_file": str(extract_path),
        "index_stats_file": str(index_stats_path),
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    summary = parse_run_summary(run_dir)
    assert summary["monitoring"]["topics_evaluated"] == 1
    assert summary["monitoring"]["events_total"] == 1
    assert summary["monitoring"]["events_by_severity"]["critical"] == 1


def test_monitor_digest_flush_and_status(tmp_path) -> None:
    queue_path = tmp_path / "runs" / "audit" / "monitor_digest_queue.json"
    queue_path.parent.mkdir(parents=True, exist_ok=True)
    queue_path.write_text(
        json.dumps(
            [
                {"run_id": "r1", "topic": "t1", "decision": {"trigger_id": "a"}},
                {"run_id": "r1", "topic": "t1", "decision": {"trigger_id": "b"}},
            ]
        ),
        encoding="utf-8",
    )

    cfg = {
        "monitoring": {"digest": {"path": str(queue_path), "enabled": True}},
        "paths": {"audit_dir": str(tmp_path / "runs" / "audit")},
    }

    sched_dir = tmp_path / "runs" / "monitor" / "schedules"
    sched_dir.mkdir(parents=True, exist_ok=True)
    (sched_dir / "finance_risk.json").write_text(
        json.dumps(
            {
                "name": "finance_risk",
                "schedule": "0 */6 * * *",
                "backend": "file",
                "updated_utc": "2026-02-24T00:00:00Z",
            }
        ),
        encoding="utf-8",
    )
    (sched_dir / "policy_watch.json").write_text(
        json.dumps(
            {
                "name": "policy_watch",
                "schedule": "0 */3 * * *",
                "backend": "crontab",
                "updated_utc": "2026-02-24T00:00:00Z",
            }
        ),
        encoding="utf-8",
    )

    old_cwd = Path.cwd()
    try:
        # monitor_status currently inspects runs/monitor/schedules relative to cwd.
        # Switch to tmp workspace for this integration check.
        import os

        os.chdir(tmp_path)
        st = monitor_status(config=cfg)
    finally:
        os.chdir(old_cwd)
    assert st["digest_queue_count"] == 2
    assert st["schedule_registry_count"] == 2
    assert st["schedule_registry_by_backend"]["file"] == 1
    assert st["schedule_registry_by_backend"]["crontab"] == 1

    out = monitor_digest_flush(config=cfg)
    assert out["queued"] == 2
    assert out["flushed"] == 2
    assert json.loads(queue_path.read_text(encoding="utf-8")) == []
