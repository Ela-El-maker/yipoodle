from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import json
import os
import sqlite3
import subprocess
import sys
import smtplib
from email.message import EmailMessage
from typing import Any

import yaml
import requests
from src.apps.monitoring_engine import evaluate_topic_monitoring


def _utc_ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def slugify(text: str) -> str:
    out = "".join(ch.lower() if ch.isalnum() else "_" for ch in text.strip())
    out = "_".join(part for part in out.split("_") if part)
    return out or "topic"


def _deep_merge(base: dict[str, Any], incoming: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for k, v in incoming.items():
        if isinstance(v, dict) and isinstance(merged.get(k), dict):
            merged[k] = _deep_merge(merged[k], v)
        else:
            merged[k] = v
    return merged


def load_automation_config(path: str) -> dict[str, Any]:
    defaults: dict[str, Any] = {
        "global": {
            "with_semantic_scholar": True,
            "prefer_arxiv": True,
            "require_pdf": False,
            "with_vector": False,
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
            "batch_size": 64,
            "min_text_chars": 200,
            "two_column_mode": "auto",
            "layout_engine": "shadow",
            "layout_promotion_state": "runs/audit/layout_promotion_state.json",
            "layout_table_handling": "linearize",
            "layout_footnote_handling": "append",
            "layout_min_region_confidence": 0.55,
            # OCR fields are optional; when omitted, extract-corpus can use sources-config OCR defaults.
            "ocr_enabled": None,
            "ocr_timeout_sec": None,
            "ocr_min_chars_trigger": None,
            "ocr_max_pages": None,
            "ocr_min_output_chars": None,
            "ocr_min_gain_chars": None,
            "ocr_min_confidence": None,
            "ocr_lang": None,
            "ocr_profile": None,
            "ocr_noise_suppression": None,
            "run_benchmark": True,
            "run_snapshot": True,
            "continue_on_topic_error": True,
            "retrieval_mode": "hybrid",
            "alpha": 0.6,
            "max_per_paper": 2,
            "quality_prior_weight": 0.15,
        },
        "paths": {
            "db_path": "data/automation/papers.db",
            "sources_config": "config/sources.yaml",
            "papers_dir": "data/automation/papers",
            "extracted_dir": "data/automation/extracted",
            "index_path": "data/indexes/automation_index.json",
            "vector_index_path": None,
            "vector_metadata_path": None,
            "reports_dir": "runs/research_reports/automation",
            "audit_dir": "runs/audit",
            "snapshot_dir": "runs/snapshots",
            "cache_dir": "runs/cache/research",
        },
        "thresholds": {
            "min_citation_coverage": 1.0,
            "warn_evidence_usage_below": 0.7,
            "fail_on_validation_error": True,
            "min_snippets": 1,
            "min_avg_chars_per_paper": 500,
            "min_avg_chars_per_page": 80,
            "max_extract_error_rate": 0.8,
            "layout_promotion_enabled": False,
            "layout_promotion_gold": None,
            "layout_promotion_min_weighted_score": 0.75,
            "layout_promotion_max_weighted_regression": 0.02,
            "layout_promotion_max_ordered_regression": 0.02,
            "layout_promotion_max_page_nonempty_regression": 0.02,
            "semantic_enabled": True,
            "semantic_mode": "offline",
            "semantic_shadow_mode": True,
            "semantic_min_support": 0.55,
            "semantic_max_contradiction": 0.30,
            "semantic_fail_on_low_support": False,
            "online_semantic_model": "gpt-4o-mini",
            "online_semantic_timeout_sec": 12.0,
            "online_semantic_max_checks": 12,
            "online_semantic_on_warn_only": True,
            "online_semantic_base_url": None,
            "online_semantic_api_key": None,
        },
        "alerts": {
            "enabled": False,
            "webhook_url": None,
            "webhook_timeout_sec": 10,
            "webhook_headers": {},
            "email_enabled": False,
            "email_to": ["comjiji7@gmail.com"],
            "email_from": "comjiji7@gmail.com",
            "smtp_host": "smtp.gmail.com",
            "smtp_port": 465,
            "smtp_use_ssl": True,
            "smtp_username": "comjiji7@gmail.com",
            "smtp_password_env": "GMAIL_APP_PASSWORD",
            "on_corpus_unhealthy": True,
            "on_topic_validation_failed": True,
            "on_source_errors": True,
        },
        "kb": {
            "enabled_default": False,
            "db_path": "data/kb/knowledge.db",
        },
        "monitoring": {
            "enabled_default": False,
            "baseline": "previous_successful_run",
            "failure_policy": "fail_open",
            "noise_defaults": {
                "cooldown_minutes": 360,
                "hysteresis_runs": 2,
            },
            "hooks": {
                "enabled": True,
                "allowlist": [],
                "timeout_sec": 5,
            },
            "digest": {
                "enabled": True,
                "window": "daily",
                "path": "runs/audit/monitor_digest_queue.json",
            },
        },
        "topics": [],
    }
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    cfg = _deep_merge(defaults, payload)
    if not cfg.get("topics"):
        raise ValueError("automation config requires at least one topic")
    return cfg


def _run_json_command(args: list[str], log_path: Path) -> dict[str, Any]:
    proc = subprocess.run(args, capture_output=True, text=True)
    stdout = proc.stdout.strip()
    stderr = proc.stderr.strip()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        json.dumps({"cmd": args, "returncode": proc.returncode, "stdout": stdout, "stderr": stderr}, indent=2),
        encoding="utf-8",
    )
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(args)}\n{stderr or stdout}")
    if not stdout:
        return {}
    try:
        return json.loads(stdout)
    except json.JSONDecodeError:
        return {"stdout": stdout}


def _run_path_command(args: list[str], log_path: Path) -> str:
    proc = subprocess.run(args, capture_output=True, text=True)
    stdout = proc.stdout.strip()
    stderr = proc.stderr.strip()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        json.dumps({"cmd": args, "returncode": proc.returncode, "stdout": stdout, "stderr": stderr}, indent=2),
        encoding="utf-8",
    )
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(args)}\n{stderr or stdout}")
    return stdout.splitlines()[-1] if stdout else ""


def _latest_run_dir(audit_dir: str) -> Path | None:
    root = Path(audit_dir) / "runs"
    if not root.exists():
        return None
    runs = sorted(
        (p for p in root.iterdir() if p.is_dir() and (p / "manifest.json").exists()),
        key=lambda p: p.name,
    )
    return runs[-1] if runs else None


def parse_run_summary(run_dir: Path) -> dict[str, Any]:
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    paths = manifest.get("paths", {})
    db_path = Path(paths.get("db_path", ""))
    papers_dir = Path(paths.get("papers_dir", ""))
    index_path = Path(paths.get("index_path", ""))

    sync_stats: list[dict[str, Any]] = []
    for item in manifest.get("topics", []):
        sync_file = item.get("sync_stats_file")
        if sync_file and Path(sync_file).exists():
            sync_stats.append(json.loads(Path(sync_file).read_text(encoding="utf-8")))

    sync_agg = {
        "fetched": sum(int(x.get("fetched", 0)) for x in sync_stats),
        "added": sum(int(x.get("added", 0)) for x in sync_stats),
        "downloaded": sum(int(x.get("downloaded", 0)) for x in sync_stats),
        "source_errors": sum(int(x.get("source_errors", 0)) for x in sync_stats),
        "missing_pdf_url": sum(int(x.get("missing_pdf_url", 0)) for x in sync_stats),
        "download_http_error": sum(int(x.get("download_http_error", 0)) for x in sync_stats),
        "non_pdf_content_type": sum(int(x.get("non_pdf_content_type", 0)) for x in sync_stats),
        "blocked_or_paywalled": sum(int(x.get("blocked_or_paywalled", 0)) for x in sync_stats),
        "require_pdf_filtered": sum(int(x.get("require_pdf_filtered", 0)) for x in sync_stats),
    }
    fetched = max(1, sync_agg["fetched"])
    sync_agg["pdf_download_rate"] = round(sync_agg["downloaded"] / fetched, 4)

    extract_stats = {}
    if manifest.get("extract_stats_file") and Path(manifest["extract_stats_file"]).exists():
        extract_stats = json.loads(Path(manifest["extract_stats_file"]).read_text(encoding="utf-8"))
        processed = int(extract_stats.get("processed", 0))
        created = int(extract_stats.get("created", 0))
        extract_stats["extract_success_rate"] = round((created / processed), 4) if processed > 0 else 0.0
    index_stats = {}
    corpus_health = {}
    if manifest.get("index_stats_file") and Path(manifest["index_stats_file"]).exists():
        index_stats = json.loads(Path(manifest["index_stats_file"]).read_text(encoding="utf-8"))
    if manifest.get("corpus_health_file") and Path(manifest["corpus_health_file"]).exists():
        corpus_health = json.loads(Path(manifest["corpus_health_file"]).read_text(encoding="utf-8"))

    benchmark = {}
    if manifest.get("benchmark_file") and Path(manifest["benchmark_file"]).exists():
        benchmark = json.loads(Path(manifest["benchmark_file"]).read_text(encoding="utf-8"))

    total_papers = 0
    if db_path.exists():
        with sqlite3.connect(db_path) as conn:
            row = conn.execute("SELECT COUNT(*) FROM papers").fetchone()
            total_papers = int(row[0] if row else 0)
    downloaded_pdfs = len(list(papers_dir.glob("*.pdf"))) if papers_dir.exists() else 0
    snippets_in_index = 0
    if index_path.exists():
        payload = json.loads(index_path.read_text(encoding="utf-8"))
        snippets_in_index = len(payload.get("snippets", []))

    topic_status: list[dict[str, Any]] = []
    monitor_events: list[dict[str, Any]] = []
    events_by_severity = {"critical": 0, "warning": 0, "info": 0}
    monitor_errors_total = 0
    for topic in manifest.get("topics", []):
        metrics = {}
        metrics_path = topic.get("metrics_path")
        if metrics_path and Path(metrics_path).exists():
            metrics = json.loads(Path(metrics_path).read_text(encoding="utf-8"))
        t_events = list(topic.get("monitor_events", []) or [])
        for ev in t_events:
            sev = str((ev or {}).get("severity", "info")).lower()
            events_by_severity[sev] = int(events_by_severity.get(sev, 0)) + 1
        monitor_events.extend(t_events)
        monitor_errors_total += int(topic.get("monitor_errors_count", 0) or 0)
        topic_status.append(
            {
                "name": topic.get("name"),
                "query": topic.get("query"),
                "report_path": topic.get("report_path"),
                "validate_ok": bool(topic.get("validate_ok", False)),
                "citation_coverage": metrics.get("citation_coverage"),
                "evidence_usage": metrics.get("evidence_usage"),
                "semantic_status": metrics.get("semantic_status"),
                "semantic_support_avg": metrics.get("semantic_support_avg"),
                "semantic_support_min": metrics.get("semantic_support_min"),
                "semantic_contradiction_max": metrics.get("semantic_contradiction_max"),
                "semantic_lines_below_threshold": metrics.get("semantic_lines_below_threshold"),
                "online_semantic_checked": metrics.get("online_semantic_checked"),
                "online_semantic_status": metrics.get("online_semantic_status"),
                "online_semantic_latency_ms": metrics.get("online_semantic_latency_ms"),
                "kb_ingest_ok": topic.get("kb_ingest_ok"),
                "kb_diff_path": topic.get("kb_diff_path"),
                "monitoring_enabled": topic.get("monitoring_enabled"),
                "monitor_baseline_run_id": topic.get("monitor_baseline_run_id"),
                "monitor_diff_path": topic.get("monitor_diff_path"),
                "monitor_trigger_results_path": topic.get("monitor_trigger_results_path"),
                "monitor_events_count": topic.get("monitor_events_count", 0),
                "monitor_errors_count": topic.get("monitor_errors_count", 0),
                "monitor_status": topic.get("monitor_status", "ok"),
            }
        )

    return {
        "run_id": manifest.get("run_id"),
        "created_utc": manifest.get("created_utc"),
        "any_topic_failure": bool(manifest.get("any_topic_failure", False)),
        "sync": sync_agg,
        "extract": extract_stats,
        "index": index_stats,
        "corpus_health": corpus_health,
        "corpus_profile": {
            "total_papers_in_db": total_papers,
            "total_downloaded_pdfs": downloaded_pdfs,
            "total_snippets_in_index": snippets_in_index,
        },
        "topics": topic_status,
        "monitoring": {
            "topics_evaluated": sum(1 for t in topic_status if bool(t.get("monitoring_enabled"))),
            "events_total": len(monitor_events),
            "events_by_severity": events_by_severity,
            "monitoring_errors": monitor_errors_total,
            "events": monitor_events,
        },
        "benchmark": benchmark,
        "snapshot_path": manifest.get("snapshot_path"),
    }


def collect_alert_events(summary: dict[str, Any], config: dict[str, Any]) -> list[dict[str, Any]]:
    alerts_cfg = (config or {}).get("alerts", {}) if isinstance(config, dict) else {}
    thresholds = (config or {}).get("thresholds", {}) if isinstance(config, dict) else {}
    events: list[dict[str, Any]] = []

    health = summary.get("corpus_health", {}) or {}
    if bool(alerts_cfg.get("on_corpus_unhealthy", True)) and health and not bool(health.get("healthy", True)):
        events.append(
            {
                "severity": "critical",
                "code": "corpus_unhealthy",
                "message": "Corpus health gate failed",
                "details": {"reasons": health.get("reasons", []), "warnings": health.get("warnings", [])},
            }
        )

    if bool(alerts_cfg.get("on_topic_validation_failed", True)):
        bad_topics = [t for t in (summary.get("topics", []) or []) if not bool(t.get("validate_ok", False))]
        if bad_topics:
            events.append(
                {
                    "severity": "critical",
                    "code": "topic_validation_failed",
                    "message": "One or more topic validations failed",
                    "details": {"topics": [{"name": t.get("name"), "report_path": t.get("report_path")} for t in bad_topics]},
                }
            )

    warn_threshold = float(thresholds.get("warn_evidence_usage_below", 0.7))
    low_evidence = [
        t
        for t in (summary.get("topics", []) or [])
        if t.get("evidence_usage") is not None and float(t.get("evidence_usage", 1.0)) < warn_threshold
    ]
    if low_evidence:
        events.append(
            {
                "severity": "warning",
                "code": "low_evidence_usage",
                "message": "Evidence usage below warning threshold for some topics",
                "details": {
                    "threshold": warn_threshold,
                    "topics": [
                        {"name": t.get("name"), "evidence_usage": t.get("evidence_usage"), "report_path": t.get("report_path")}
                        for t in low_evidence
                    ],
                },
            }
        )

    if bool(alerts_cfg.get("on_source_errors", True)):
        source_errors = int((summary.get("sync", {}) or {}).get("source_errors", 0))
        if source_errors > 0:
            events.append(
                {
                    "severity": "warning",
                    "code": "source_errors",
                    "message": "Source connector errors occurred during sync",
                    "details": {"source_errors": source_errors},
                }
            )

    if bool(summary.get("any_topic_failure", False)):
        events.append(
            {
                "severity": "critical",
                "code": "topic_runtime_failure",
                "message": "At least one topic raised runtime failure during automation run",
                "details": {},
            }
        )

    monitoring = (summary.get("monitoring", {}) if isinstance(summary, dict) else {}) or {}
    for ev in monitoring.get("events", []) or []:
        code = str((ev or {}).get("code", "")).strip()
        if not code:
            continue
        if code == "monitor_trigger_medium_digest_queued":
            # medium severity routes via digest queue, not immediate alerting
            continue
        events.append(ev)
    return events


def dispatch_alerts(summary: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    alerts_cfg = (config or {}).get("alerts", {}) if isinstance(config, dict) else {}
    enabled = bool(alerts_cfg.get("enabled", False))
    events = collect_alert_events(summary, config)
    result: dict[str, Any] = {"enabled": enabled, "events_count": len(events), "sent": False, "transport": None}
    if not enabled or not events:
        return result

    payload = {
        "source": "yipoodle-automation",
        "run_id": summary.get("run_id"),
        "created_utc": summary.get("created_utc"),
        "events": events,
        "summary_path": "runs/audit/latest_summary.json",
    }
    transports: list[dict[str, Any]] = []

    # Webhook transport (optional)
    url = alerts_cfg.get("webhook_url")
    if url:
        headers = dict(alerts_cfg.get("webhook_headers", {}) or {})
        timeout = float(alerts_cfg.get("webhook_timeout_sec", 10))
        tr: dict[str, Any] = {"name": "webhook", "sent": False}
        try:
            resp = requests.post(str(url), json=payload, headers=headers, timeout=timeout)
            tr["status_code"] = int(resp.status_code)
            if resp.status_code >= 300:
                tr["error"] = f"webhook_http_{resp.status_code}"
            else:
                tr["sent"] = True
        except Exception as exc:
            tr["error"] = str(exc)
        transports.append(tr)

    # Email transport (optional; Gmail-compatible SMTP defaults)
    if bool(alerts_cfg.get("email_enabled", False)):
        tr = {"name": "email", "sent": False}
        try:
            recipients = alerts_cfg.get("email_to", []) or []
            if isinstance(recipients, str):
                recipients = [recipients]
            recipients = [str(x).strip() for x in recipients if str(x).strip()]
            if not recipients:
                tr["error"] = "email_to_missing"
                transports.append(tr)
            else:
                smtp_host = str(alerts_cfg.get("smtp_host", "smtp.gmail.com"))
                smtp_port = int(alerts_cfg.get("smtp_port", 465))
                use_ssl = bool(alerts_cfg.get("smtp_use_ssl", True))
                username = str(
                    alerts_cfg.get("smtp_username")
                    or os.getenv("GMAIL_USER")
                    or os.getenv("SMTP_USERNAME")
                    or ""
                ).strip()
                pwd_env = str(alerts_cfg.get("smtp_password_env", "GMAIL_APP_PASSWORD"))
                password = str(os.getenv(pwd_env, "")).strip()
                if not username:
                    tr["error"] = "smtp_username_missing"
                    transports.append(tr)
                elif not password:
                    tr["error"] = f"smtp_password_missing_env:{pwd_env}"
                    transports.append(tr)
                else:
                    sender = str(alerts_cfg.get("email_from") or username).strip()
                    max_sev = "warning"
                    if any(str(e.get("severity", "")).lower() == "critical" for e in events):
                        max_sev = "critical"
                    subject = f"[Yipoodle][{max_sev.upper()}] automation run {summary.get('run_id')}"
                    body = json.dumps(payload, indent=2)
                    msg = EmailMessage()
                    msg["Subject"] = subject
                    msg["From"] = sender
                    msg["To"] = ", ".join(recipients)
                    msg.set_content(body)
                    timeout = float(alerts_cfg.get("webhook_timeout_sec", 10))
                    if use_ssl:
                        with smtplib.SMTP_SSL(smtp_host, smtp_port, timeout=timeout) as server:
                            server.login(username, password)
                            server.send_message(msg)
                    else:
                        with smtplib.SMTP(smtp_host, smtp_port, timeout=timeout) as server:
                            server.starttls()
                            server.login(username, password)
                            server.send_message(msg)
                    tr["sent"] = True
                    tr["recipients"] = recipients
        except Exception as exc:
            tr["error"] = str(exc)
        transports.append(tr)

    # Backward-compatible top-level summary
    result["transports"] = transports
    result["sent"] = any(bool(t.get("sent", False)) for t in transports)
    if transports:
        result["transport"] = ",".join(str(t.get("name")) for t in transports)
    else:
        result["error"] = "alerts_enabled_but_no_transport_configured"
    if not result["sent"]:
        errors = [str(t.get("error")) for t in transports if t.get("error")]
        if errors:
            result["error"] = "; ".join(errors)
    return result


def render_summary_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Automation Run Summary",
        "",
        f"- run_id: `{summary.get('run_id')}`",
        f"- created_utc: `{summary.get('created_utc')}`",
        "",
        "## Sync",
    ]
    sync = summary.get("sync", {})
    for k in [
        "fetched",
        "added",
        "downloaded",
        "pdf_download_rate",
        "source_errors",
        "missing_pdf_url",
        "download_http_error",
        "non_pdf_content_type",
        "blocked_or_paywalled",
        "require_pdf_filtered",
    ]:
        lines.append(f"- {k}: {sync.get(k)}")

    lines.extend(["", "## Corpus Profile"])
    profile = summary.get("corpus_profile", {})
    for k in ["total_papers_in_db", "total_downloaded_pdfs", "total_snippets_in_index"]:
        lines.append(f"- {k}: {profile.get(k)}")

    lines.extend(["", "## Topics"])
    for t in summary.get("topics", []):
        lines.append(
            f"- `{t.get('name')}`: validate_ok={t.get('validate_ok')}, "
            f"citation_coverage={t.get('citation_coverage')}, evidence_usage={t.get('evidence_usage')}, "
            f"semantic_status={t.get('semantic_status')}, kb_ingest_ok={t.get('kb_ingest_ok')}"
        )
    lines.extend(["", "## Semantic Faithfulness"])
    for t in summary.get("topics", []):
        if t.get("semantic_status") is None:
            continue
        lines.append(
            f"- `{t.get('name')}`: status={t.get('semantic_status')}, "
            f"support_avg={t.get('semantic_support_avg')}, "
            f"support_min={t.get('semantic_support_min')}, "
            f"contradiction_max={t.get('semantic_contradiction_max')}, "
            f"lines_below_threshold={t.get('semantic_lines_below_threshold')}, "
            f"online_checked={t.get('online_semantic_checked')}, "
            f"online_status={t.get('online_semantic_status')}, "
            f"online_latency_ms={t.get('online_semantic_latency_ms')}"
        )
    lines.extend(["", "## Extract"])
    extract = summary.get("extract", {})
    for k in [
        "extract_success_rate",
        "two_column_applied_count",
        "layout_v2_attempted_count",
        "layout_v2_applied_count",
        "layout_shadow_compared_count",
        "layout_shadow_diff_rate",
        "layout_confidence_avg",
        "layout_fallback_to_legacy_count",
        "ocr_attempted_count",
        "ocr_succeeded_count",
        "ocr_failed_count",
        "ocr_rejected_low_quality_count",
        "ocr_rejected_low_confidence_count",
        "ocr_avg_confidence",
        "quality_score_avg",
        "quality_score_min",
        "quality_score_max",
    ]:
        if k in extract:
            lines.append(f"- {k}: {extract.get(k)}")
    if "quality_band_counts" in extract:
        lines.append(f"- quality_band_counts: {extract.get('quality_band_counts')}")
    if "layout_region_counts" in extract:
        lines.append(f"- layout_region_counts: {extract.get('layout_region_counts')}")
    lines.extend(["", "## Corpus Health"])
    health = summary.get("corpus_health", {})
    for k in [
        "healthy",
        "paper_count",
        "papers_with_page_stats",
        "page_stats_coverage_pct",
        "snippet_count",
        "total_pages",
        "avg_chars_per_paper",
        "avg_chars_per_page",
        "avg_chars_per_snippet",
        "extract_error_rate",
    ]:
        if k in health:
            lines.append(f"- {k}: {health.get(k)}")
    if health.get("warnings"):
        lines.append(f"- warnings: {health.get('warnings')}")
    if health.get("reasons"):
        lines.append(f"- reasons: {health.get('reasons')}")
    lines.extend(["", "## Monitoring"])
    mon = summary.get("monitoring", {}) or {}
    lines.append(f"- topics_evaluated: {mon.get('topics_evaluated', 0)}")
    lines.append(f"- events_total: {mon.get('events_total', 0)}")
    lines.append(f"- events_by_severity: {mon.get('events_by_severity', {})}")
    lines.append(f"- monitoring_errors: {mon.get('monitoring_errors', 0)}")
    return "\n".join(lines) + "\n"


def write_summary_outputs(
    *,
    audit_dir: str,
    run_dir: str | None = None,
    out_json: str = "runs/audit/latest_summary.json",
    out_md: str = "runs/audit/latest_summary.md",
) -> dict[str, Any]:
    target = Path(run_dir) if run_dir else _latest_run_dir(audit_dir)
    if target is None or not target.exists():
        raise FileNotFoundError("No automation run directory found")
    summary = parse_run_summary(target)
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(out_json).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    Path(out_md).write_text(render_summary_markdown(summary), encoding="utf-8")

    history_path = Path(audit_dir) / "pdf_download_rate_history.json"
    if history_path.exists():
        history = json.loads(history_path.read_text(encoding="utf-8"))
    else:
        history = []
    run_id = summary.get("run_id")
    if run_id and all(entry.get("run_id") != run_id for entry in history):
        history.append({"run_id": run_id, "pdf_download_rate": summary.get("sync", {}).get("pdf_download_rate")})
        history_path.write_text(json.dumps(history[-52:], indent=2), encoding="utf-8")
    return summary


def run_automation(config_path: str) -> str:
    cfg = load_automation_config(config_path)
    run_id = _utc_ts()
    audit_dir = Path(cfg["paths"]["audit_dir"])
    run_dir = audit_dir / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    py = str(Path(sys.executable))
    cli = ["-m", "src.cli"]
    global_cfg = cfg["global"]
    paths = cfg["paths"]
    thresholds = cfg["thresholds"]

    topics_manifest: list[dict[str, Any]] = []
    any_topic_failure = False

    for topic in cfg["topics"]:
        name = topic["name"]
        slug = slugify(name)
        query = topic["query"]
        max_results = int(topic.get("max_results", 20))
        sync_log = run_dir / f"sync_{slug}.log.json"
        sync_stats = _run_json_command(
            [
                py,
                *cli,
                "sync-papers",
                "--query",
                query,
                "--max-results",
                str(max_results),
                "--db-path",
                paths["db_path"],
                "--papers-dir",
                paths["papers_dir"],
                *(["--sources-config", str(paths["sources_config"])] if paths.get("sources_config") else []),
                *([] if not global_cfg.get("with_semantic_scholar", False) else ["--with-semantic-scholar"]),
                *([] if not global_cfg.get("prefer_arxiv", False) else ["--prefer-arxiv"]),
                *([] if not global_cfg.get("require_pdf", False) else ["--require-pdf"]),
            ],
            sync_log,
        )
        sync_stats_path = run_dir / f"sync_{slug}.stats.json"
        sync_stats_path.write_text(json.dumps(sync_stats, indent=2), encoding="utf-8")
        topics_manifest.append(
            {
                "name": name,
                "slug": slug,
                "query": query,
                "sync_stats_file": str(sync_stats_path),
                "validate_ok": False,
            }
        )

    extract_stats_path = run_dir / "extract.stats.json"
    if bool(thresholds.get("layout_promotion_enabled", False)) and thresholds.get("layout_promotion_gold"):
        _run_json_command(
            [
                py,
                *cli,
                "layout-promotion-gate",
                "--papers-dir",
                paths["papers_dir"],
                "--gold",
                str(thresholds.get("layout_promotion_gold")),
                "--state-path",
                str(global_cfg.get("layout_promotion_state", "runs/audit/layout_promotion_state.json")),
                *(["--db-path", paths["db_path"]] if paths.get("db_path") else []),
                "--min-text-chars",
                str(int(global_cfg.get("min_text_chars", 200))),
                "--two-column-mode",
                str(global_cfg.get("two_column_mode", "auto")),
                "--min-weighted-score",
                str(float(thresholds.get("layout_promotion_min_weighted_score", 0.75))),
                "--max-weighted-regression",
                str(float(thresholds.get("layout_promotion_max_weighted_regression", 0.02))),
                "--max-ordered-regression",
                str(float(thresholds.get("layout_promotion_max_ordered_regression", 0.02))),
                "--max-page-nonempty-regression",
                str(float(thresholds.get("layout_promotion_max_page_nonempty_regression", 0.02))),
            ],
            run_dir / "layout_promotion.log.json",
        )

    extract_cmd = [
        py,
        *cli,
        "extract-corpus",
        "--papers-dir",
        paths["papers_dir"],
        "--out-dir",
        paths["extracted_dir"],
        "--db-path",
        paths["db_path"],
        "--min-text-chars",
        str(int(global_cfg.get("min_text_chars", 200))),
        "--two-column-mode",
        str(global_cfg.get("two_column_mode", "auto")),
        "--layout-engine",
        str(global_cfg.get("layout_engine", "shadow")),
        "--layout-promotion-state",
        str(global_cfg.get("layout_promotion_state", "runs/audit/layout_promotion_state.json")),
        "--layout-table-handling",
        str(global_cfg.get("layout_table_handling", "linearize")),
        "--layout-footnote-handling",
        str(global_cfg.get("layout_footnote_handling", "append")),
        "--layout-min-region-confidence",
        str(float(global_cfg.get("layout_min_region_confidence", 0.55))),
        *(["--sources-config", str(paths["sources_config"])] if paths.get("sources_config") else []),
    ]
    if global_cfg.get("ocr_enabled") is not None:
        extract_cmd.extend(["--ocr-enabled"] if bool(global_cfg.get("ocr_enabled")) else ["--no-ocr-enabled"])
    if global_cfg.get("ocr_timeout_sec") is not None:
        extract_cmd.extend(["--ocr-timeout-sec", str(int(global_cfg.get("ocr_timeout_sec")))])
    if global_cfg.get("ocr_min_chars_trigger") is not None:
        extract_cmd.extend(["--ocr-min-chars-trigger", str(int(global_cfg.get("ocr_min_chars_trigger")))])
    if global_cfg.get("ocr_max_pages") is not None:
        extract_cmd.extend(["--ocr-max-pages", str(int(global_cfg.get("ocr_max_pages")))])
    if global_cfg.get("ocr_min_output_chars") is not None:
        extract_cmd.extend(["--ocr-min-output-chars", str(int(global_cfg.get("ocr_min_output_chars")))])
    if global_cfg.get("ocr_min_gain_chars") is not None:
        extract_cmd.extend(["--ocr-min-gain-chars", str(int(global_cfg.get("ocr_min_gain_chars")))])
    if global_cfg.get("ocr_min_confidence") is not None:
        extract_cmd.extend(["--ocr-min-confidence", str(float(global_cfg.get("ocr_min_confidence")))])
    if global_cfg.get("ocr_lang") is not None:
        extract_cmd.extend(["--ocr-lang", str(global_cfg.get("ocr_lang"))])
    if global_cfg.get("ocr_profile") is not None:
        extract_cmd.extend(["--ocr-profile", str(global_cfg.get("ocr_profile"))])
    if global_cfg.get("ocr_noise_suppression") is not None:
        extract_cmd.extend(
            ["--ocr-noise-suppression"] if bool(global_cfg.get("ocr_noise_suppression")) else ["--no-ocr-noise-suppression"]
        )

    extract_stats = _run_json_command(extract_cmd, run_dir / "extract.log.json")
    extract_stats_path.write_text(json.dumps(extract_stats, indent=2), encoding="utf-8")

    corpus_health_stats_path = run_dir / "corpus_health.stats.json"
    corpus_health_stats = _run_json_command(
        [
            py,
            *cli,
            "corpus-health",
            "--corpus",
            paths["extracted_dir"],
            "--extract-stats",
            str(extract_stats_path),
            "--min-snippets",
            str(int(thresholds.get("min_snippets", 1))),
            "--min-avg-chars-per-paper",
            str(int(thresholds.get("min_avg_chars_per_paper", 500))),
            "--min-avg-chars-per-page",
            str(int(thresholds.get("min_avg_chars_per_page", 80))),
            "--max-extract-error-rate",
            str(float(thresholds.get("max_extract_error_rate", 0.8))),
        ],
        run_dir / "corpus_health.log.json",
    )
    corpus_health_stats_path.write_text(json.dumps(corpus_health_stats, indent=2), encoding="utf-8")
    if not bool(corpus_health_stats.get("healthy", False)):
        raise RuntimeError("Corpus health check failed before index build")

    index_stats_path = run_dir / "index.stats.json"
    index_stats = _run_json_command(
        [
            py,
            *cli,
            "build-index",
            "--corpus",
            paths["extracted_dir"],
            "--db-path",
            paths["db_path"],
            "--out",
            paths["index_path"],
            *([] if not global_cfg.get("with_vector", False) else ["--with-vector"]),
            "--embedding-model",
            str(global_cfg.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2")),
            "--batch-size",
            str(int(global_cfg.get("batch_size", 64))),
            "--vector-index-type",
            str(global_cfg.get("vector_index_type", "flat")),
            "--vector-nlist",
            str(int(global_cfg.get("vector_nlist", 1024))),
            "--vector-m",
            str(int(global_cfg.get("vector_m", 32))),
            "--vector-ef-construction",
            str(int(global_cfg.get("vector_ef_construction", 200))),
            "--vector-shards",
            str(int(global_cfg.get("vector_shards", 1))),
            "--vector-train-sample-size",
            str(int(global_cfg.get("vector_train_sample_size", 200000))),
            *(["--vector-index-path", str(paths["vector_index_path"])] if paths.get("vector_index_path") else []),
            *(["--vector-metadata-path", str(paths["vector_metadata_path"])] if paths.get("vector_metadata_path") else []),
        ],
        run_dir / "index.log.json",
    )
    index_stats_path.write_text(json.dumps(index_stats, indent=2), encoding="utf-8")

    reports_dir = Path(paths["reports_dir"]) / run_id
    reports_dir.mkdir(parents=True, exist_ok=True)

    for topic_state, topic in zip(topics_manifest, cfg["topics"]):
        slug = topic_state["slug"]
        query = topic_state["query"]
        top_k = int(topic.get("top_k", 8))
        min_items = int(topic.get("min_items", 2))
        min_score = float(topic.get("min_score", 0.5))
        retrieval_mode = str(topic.get("retrieval_mode", global_cfg.get("retrieval_mode", "lexical")))
        alpha = float(topic.get("alpha", global_cfg.get("alpha", 0.6)))
        max_per_paper = int(topic.get("max_per_paper", global_cfg.get("max_per_paper", 2)))
        report_path = reports_dir / f"{slug}.md"
        topic_state["report_path"] = str(report_path)
        try:
            _ = _run_path_command(
                [
                    py,
                    *cli,
                    "research",
                    "--index",
                    paths["index_path"],
                    "--question",
                    query,
                    "--top-k",
                    str(top_k),
                    "--min-items",
                    str(min_items),
                    "--min-score",
                    str(min_score),
                    "--retrieval-mode",
                    retrieval_mode,
                    "--alpha",
                    str(alpha),
                    "--max-per-paper",
                    str(max_per_paper),
                    "--cache-dir",
                    paths["cache_dir"],
                    "--embedding-model",
                    str(global_cfg.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2")),
                    "--quality-prior-weight",
                    str(float(topic.get("quality_prior_weight", global_cfg.get("quality_prior_weight", 0.15)))),
                    *(
                        ["--vector-service-endpoint", str(global_cfg.get("vector_service_endpoint"))]
                        if global_cfg.get("vector_service_endpoint")
                        else []
                    ),
                    "--vector-nprobe",
                    str(int(topic.get("vector_nprobe", global_cfg.get("vector_nprobe", 16)))),
                    "--vector-ef-search",
                    str(int(topic.get("vector_ef_search", global_cfg.get("vector_ef_search", 64)))),
                    "--vector-topk-candidate-multiplier",
                    str(float(topic.get("vector_topk_candidate_multiplier", global_cfg.get("vector_topk_candidate_multiplier", 1.5)))),
                    "--semantic-mode",
                    str(thresholds.get("semantic_mode", "offline")),
                    "--semantic-model",
                    str(global_cfg.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2")),
                    "--semantic-min-support",
                    str(float(thresholds.get("semantic_min_support", 0.55))),
                    "--semantic-max-contradiction",
                    str(float(thresholds.get("semantic_max_contradiction", 0.30))),
                    *(
                        ["--semantic-shadow-mode"]
                        if bool(thresholds.get("semantic_shadow_mode", True))
                        else ["--no-semantic-shadow-mode"]
                    ),
                    *(
                        ["--semantic-fail-on-low-support"]
                        if bool(thresholds.get("semantic_fail_on_low_support", False))
                        else []
                    ),
                    "--online-semantic-model",
                    str(thresholds.get("online_semantic_model", "gpt-4o-mini")),
                    "--online-semantic-timeout-sec",
                    str(float(thresholds.get("online_semantic_timeout_sec", 12.0))),
                    "--online-semantic-max-checks",
                    str(int(thresholds.get("online_semantic_max_checks", 12))),
                    *(
                        ["--online-semantic-on-warn-only"]
                        if bool(thresholds.get("online_semantic_on_warn_only", True))
                        else ["--no-online-semantic-on-warn-only"]
                    ),
                    *(
                        ["--online-semantic-base-url", str(thresholds.get("online_semantic_base_url"))]
                        if thresholds.get("online_semantic_base_url")
                        else []
                    ),
                    *(
                        ["--online-semantic-api-key", str(thresholds.get("online_semantic_api_key"))]
                        if thresholds.get("online_semantic_api_key")
                        else []
                    ),
                    *(["--sources-config", str(paths["sources_config"])] if paths.get("sources_config") else []),
                    *(["--vector-index-path", str(paths["vector_index_path"])] if paths.get("vector_index_path") else []),
                    *(["--vector-metadata-path", str(paths["vector_metadata_path"])] if paths.get("vector_metadata_path") else []),
                    "--out",
                    str(report_path),
                ],
                run_dir / f"research_{slug}.log.json",
            )
            _run_path_command(
                [
                    py,
                    *cli,
                    "validate-report",
                    "--input",
                    str(report_path),
                    "--evidence",
                    str(report_path.with_suffix(".evidence.json")),
                    *(
                        ["--semantic-faithfulness"]
                        if bool(thresholds.get("semantic_enabled", True))
                        else ["--no-semantic-faithfulness"]
                    ),
                    "--semantic-mode",
                    str(thresholds.get("semantic_mode", "offline")),
                    *(
                        ["--semantic-shadow-mode"]
                        if bool(thresholds.get("semantic_shadow_mode", True))
                        else ["--no-semantic-shadow-mode"]
                    ),
                    "--semantic-min-support",
                    str(float(thresholds.get("semantic_min_support", 0.55))),
                    "--semantic-max-contradiction",
                    str(float(thresholds.get("semantic_max_contradiction", 0.30))),
                    *(
                        ["--semantic-fail-on-low-support"]
                        if bool(thresholds.get("semantic_fail_on_low_support", False))
                        else []
                    ),
                    "--online-semantic-model",
                    str(thresholds.get("online_semantic_model", "gpt-4o-mini")),
                    "--online-semantic-timeout-sec",
                    str(float(thresholds.get("online_semantic_timeout_sec", 12.0))),
                    "--online-semantic-max-checks",
                    str(int(thresholds.get("online_semantic_max_checks", 12))),
                    *(
                        ["--online-semantic-on-warn-only"]
                        if bool(thresholds.get("online_semantic_on_warn_only", True))
                        else ["--no-online-semantic-on-warn-only"]
                    ),
                    *(
                        ["--online-semantic-base-url", str(thresholds.get("online_semantic_base_url"))]
                        if thresholds.get("online_semantic_base_url")
                        else []
                    ),
                    *(
                        ["--online-semantic-api-key", str(thresholds.get("online_semantic_api_key"))]
                        if thresholds.get("online_semantic_api_key")
                        else []
                    ),
                ],
                run_dir / f"validate_{slug}.log.json",
            )
            metrics_path = report_path.with_suffix(".metrics.json")
            topic_state["metrics_path"] = str(metrics_path)
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            topic_state["validate_ok"] = bool(metrics.get("citation_coverage", 0.0) >= thresholds["min_citation_coverage"])
            topic_state["warn_low_evidence_usage"] = bool(
                float(metrics.get("evidence_usage", 0.0)) < float(thresholds["warn_evidence_usage_below"])
            )

            kb_cfg = cfg.get("kb", {}) if isinstance(cfg, dict) else {}
            kb_enabled = bool(topic.get("kb_enabled", kb_cfg.get("enabled_default", False)))
            if kb_enabled and topic_state.get("validate_ok", False):
                kb_db = str(topic.get("kb_db", kb_cfg.get("db_path", "data/kb/knowledge.db")))
                kb_topic = str(topic.get("kb_topic", topic.get("name", slug)))
                kb_ingest_stats = _run_json_command(
                    [
                        py,
                        *cli,
                        "kb-ingest",
                        "--report",
                        str(report_path),
                        "--evidence",
                        str(report_path.with_suffix(".evidence.json")),
                        "--metrics",
                        str(report_path.with_suffix(".metrics.json")),
                        "--kb-db",
                        kb_db,
                        "--topic",
                        kb_topic,
                        "--run-id",
                        run_id,
                    ],
                    run_dir / f"kb_ingest_{slug}.log.json",
                )
                kb_ingest_path = run_dir / f"kb_ingest_{slug}.stats.json"
                kb_ingest_path.write_text(json.dumps(kb_ingest_stats, indent=2), encoding="utf-8")
                topic_state["kb_ingest_ok"] = bool(kb_ingest_stats.get("kb_ingest_succeeded", False))
                topic_state["kb_ingest_path"] = str(kb_ingest_path)

                if bool(topic.get("kb_diff_alert", False)):
                    since_run = str(topic.get("kb_since_run", ""))
                    kb_diff_stats = _run_json_command(
                        [
                            py,
                            *cli,
                            "kb-diff",
                            "--kb-db",
                            kb_db,
                            "--topic",
                            kb_topic,
                            *(["--since-run", since_run] if since_run else []),
                        ],
                        run_dir / f"kb_diff_{slug}.log.json",
                    )
                    kb_diff_path = run_dir / f"kb_diff_{slug}.json"
                    kb_diff_path.write_text(json.dumps(kb_diff_stats, indent=2), encoding="utf-8")
                    topic_state["kb_diff_path"] = str(kb_diff_path)

            mon_stats = evaluate_topic_monitoring(
                cfg=cfg,
                run_id=run_id,
                run_dir=str(run_dir),
                topic=topic,
                topic_state=topic_state,
            )
            topic_state.update(mon_stats)
        except Exception as exc:
            topic_state["error"] = str(exc)
            any_topic_failure = True
            if not global_cfg.get("continue_on_topic_error", True):
                raise

    benchmark_file = ""
    if global_cfg.get("run_benchmark", True):
        benchmark_file = str((reports_dir / "benchmark.json").resolve())
        _run_path_command(
            [
                py,
                *cli,
                "benchmark-research",
                "--index",
                paths["index_path"],
                "--queries-file",
                "tests/fixtures/queries.txt",
                "--runs-per-query",
                "2",
                "--retrieval-mode",
                str(global_cfg.get("retrieval_mode", "lexical")),
                "--alpha",
                str(float(global_cfg.get("alpha", 0.6))),
                "--max-per-paper",
                str(int(global_cfg.get("max_per_paper", 2))),
                "--embedding-model",
                str(global_cfg.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2")),
                "--quality-prior-weight",
                str(float(global_cfg.get("quality_prior_weight", 0.15))),
                *(
                    ["--vector-service-endpoint", str(global_cfg.get("vector_service_endpoint"))]
                    if global_cfg.get("vector_service_endpoint")
                    else []
                ),
                "--vector-nprobe",
                str(int(global_cfg.get("vector_nprobe", 16))),
                "--vector-ef-search",
                str(int(global_cfg.get("vector_ef_search", 64))),
                "--vector-topk-candidate-multiplier",
                str(float(global_cfg.get("vector_topk_candidate_multiplier", 1.5))),
                *(["--sources-config", str(paths["sources_config"])] if paths.get("sources_config") else []),
                *(["--vector-index-path", str(paths["vector_index_path"])] if paths.get("vector_index_path") else []),
                *(["--vector-metadata-path", str(paths["vector_metadata_path"])] if paths.get("vector_metadata_path") else []),
                "--out",
                benchmark_file,
            ],
            run_dir / "benchmark.log.json",
        )

    snapshot_path = ""
    if global_cfg.get("run_snapshot", True):
        first_report = next((t.get("report_path") for t in topics_manifest if t.get("report_path")), None)
        if first_report:
            snapshot_path = _run_path_command(
                [
                    py,
                    *cli,
                    "snapshot-run",
                    "--report",
                    first_report,
                    "--index",
                    paths["index_path"],
                    "--config",
                    "config/train.yaml",
                    "--config",
                    "config/sources.yaml",
                    "--config",
                    config_path,
                    "--out",
                    paths["snapshot_dir"],
                ],
                run_dir / "snapshot.log.json",
            )

    manifest = {
        "run_id": run_id,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": str(Path(config_path).resolve()),
        "paths": {
            "db_path": str(Path(paths["db_path"]).resolve()),
            "papers_dir": str(Path(paths["papers_dir"]).resolve()),
            "extracted_dir": str(Path(paths["extracted_dir"]).resolve()),
            "index_path": str(Path(paths["index_path"]).resolve()),
            "reports_dir": str(reports_dir.resolve()),
        },
        "topics": topics_manifest,
        "extract_stats_file": str(extract_stats_path.resolve()),
        "corpus_health_file": str(corpus_health_stats_path.resolve()),
        "index_stats_file": str(index_stats_path.resolve()),
        "benchmark_file": benchmark_file,
        "snapshot_path": snapshot_path,
        "any_topic_failure": any_topic_failure,
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    summary = write_summary_outputs(
        audit_dir=paths["audit_dir"],
        run_dir=str(run_dir),
        out_json=str(audit_dir / "latest_summary.json"),
        out_md=str(audit_dir / "latest_summary.md"),
    )

    if thresholds.get("fail_on_validation_error", True):
        if any_topic_failure or any(not bool(t.get("validate_ok")) for t in summary.get("topics", [])):
            raise RuntimeError("Automation run has topic failures or validation threshold failures")
    return str(run_dir)
