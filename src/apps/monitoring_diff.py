from __future__ import annotations

from pathlib import Path
import json
from typing import Any


def _load_json(path: str | Path | None) -> dict[str, Any]:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _load_text(path: str | Path | None) -> str:
    if not path:
        return ""
    p = Path(path)
    if not p.exists():
        return ""
    try:
        return p.read_text(encoding="utf-8")
    except Exception:
        return ""


def _evidence_ids(report_path: str | None) -> set[str]:
    if not report_path:
        return set()
    ep = Path(report_path).with_suffix(".evidence.json")
    payload = _load_json(ep)
    items = payload.get("items", []) if isinstance(payload, dict) else []
    ids: set[str] = set()
    for it in items:
        sid = str((it or {}).get("snippet_id", "")).strip()
        if sid:
            ids.add(sid)
    return ids


def _split_report_sections(text: str) -> dict[str, str]:
    if not text:
        return {"full_report": "", "synthesis": "", "gaps": ""}
    lines = text.splitlines()
    synthesis: list[str] = []
    gaps: list[str] = []
    cur = "synthesis"
    for ln in lines:
        s = ln.strip()
        if s.lower().startswith("## gaps"):
            cur = "gaps"
            continue
        if s.startswith("## ") and not s.lower().startswith("## synthesis"):
            cur = "other"
            continue
        if cur == "synthesis":
            synthesis.append(ln)
        elif cur == "gaps":
            gaps.append(ln)
    return {
        "full_report": text,
        "synthesis": "\n".join(synthesis),
        "gaps": "\n".join(gaps),
    }


def find_previous_successful_topic_state(*, audit_dir: str, current_run_id: str, topic_name: str) -> tuple[str | None, dict[str, Any] | None]:
    runs_root = Path(audit_dir) / "runs"
    if not runs_root.exists():
        return None, None
    candidates = sorted([p for p in runs_root.iterdir() if p.is_dir() and p.name < current_run_id], key=lambda p: p.name, reverse=True)
    for run_dir in candidates:
        manifest_path = run_dir / "manifest.json"
        if not manifest_path.exists():
            continue
        manifest = _load_json(manifest_path)
        for t in manifest.get("topics", []) or []:
            if str(t.get("name")) != topic_name:
                continue
            if bool(t.get("validate_ok", False)) and not str(t.get("error", "")).strip():
                return str(manifest.get("run_id", run_dir.name)), t
    return None, None


def build_monitor_delta(
    *,
    run_id: str,
    topic_name: str,
    topic_state: dict[str, Any],
    baseline_run_id: str | None,
    baseline_topic_state: dict[str, Any] | None,
) -> dict[str, Any]:
    cur_metrics = _load_json(topic_state.get("metrics_path"))
    base_metrics = _load_json((baseline_topic_state or {}).get("metrics_path"))

    cur_ids = _evidence_ids(topic_state.get("report_path"))
    base_ids = _evidence_ids((baseline_topic_state or {}).get("report_path"))
    new_ids = sorted(cur_ids - base_ids)

    kb_diff = _load_json(topic_state.get("kb_diff_path"))
    counts = kb_diff.get("aggregate_counts", {}) if isinstance(kb_diff, dict) else {}

    text = _load_text(topic_state.get("report_path"))
    report_text = _split_report_sections(text)

    return {
        "run_id": run_id,
        "topic": topic_name,
        "baseline_run_id": baseline_run_id,
        "current_report_path": topic_state.get("report_path"),
        "baseline_report_path": (baseline_topic_state or {}).get("report_path"),
        "current_sources_count": len(cur_ids),
        "baseline_sources_count": len(base_ids),
        "new_sources_count": len(new_ids),
        "new_source_ids": new_ids,
        "metrics_current": cur_metrics,
        "metrics_baseline": base_metrics,
        "kb_diff_counts": {
            "added": int((counts or {}).get("added", 0)),
            "updated": int((counts or {}).get("updated", 0)),
            "disputed": int((counts or {}).get("disputed", 0)),
            "superseded": int((counts or {}).get("superseded", 0)),
        },
        "report_text": report_text,
    }
