from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import json
from typing import Any

from .monitoring_diff import build_monitor_delta, find_previous_successful_topic_state
from .monitoring_hooks import run_monitor_hooks
from .monitoring_rules import evaluate_monitor_rules
from .monitoring_state import apply_noise_controls, load_monitor_state, save_monitor_state


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _slug(text: str) -> str:
    out = "".join(ch.lower() if ch.isalnum() else "_" for ch in text.strip())
    out = "_".join(part for part in out.split("_") if part)
    return out or "topic"


def _write_json(path: Path, payload: dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return str(path)


def _monitoring_cfg(
    cfg: dict[str, Any], topic: dict[str, Any]
) -> tuple[bool, dict[str, Any], dict[str, Any], dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    mon_global = (cfg.get("monitoring", {}) if isinstance(cfg, dict) else {}) or {}
    enabled_default = bool(mon_global.get("enabled_default", False))
    enabled = bool(topic.get("monitoring_enabled", enabled_default))
    topic_mon = (topic.get("monitoring", {}) if isinstance(topic, dict) else {}) or {}
    noise_defaults = (mon_global.get("noise_defaults", {}) if isinstance(mon_global, dict) else {}) or {}
    hooks_global = (mon_global.get("hooks", {}) if isinstance(mon_global, dict) else {}) or {}
    triggers = list(topic_mon.get("triggers", []) or [])

    if not triggers and bool(topic.get("kb_diff_alert", False)):
        triggers = [
            {
                "id": "kb_diff_alert_default",
                "type": "kb_diff_count",
                "severity": "high",
                "min_added_claims": 1,
            }
        ]

    hooks = list(topic_mon.get("hooks", []) or [])
    return enabled, mon_global, noise_defaults, hooks_global, triggers, hooks


def evaluate_topic_monitoring(
    *,
    cfg: dict[str, Any],
    run_id: str,
    run_dir: str,
    topic: dict[str, Any],
    topic_state: dict[str, Any],
) -> dict[str, Any]:
    enabled, mon_global, noise_defaults, hooks_global, triggers, hooks = _monitoring_cfg(cfg, topic)
    if not enabled:
        return {
            "monitoring_enabled": False,
            "monitor_status": "ok",
            "monitor_events": [],
            "monitor_errors": [],
        }

    audit_dir = str((cfg.get("paths", {}) or {}).get("audit_dir", "runs/audit"))
    slug = _slug(str(topic_state.get("name", topic.get("name", "topic"))))

    baseline_run_id, baseline_topic_state = find_previous_successful_topic_state(
        audit_dir=audit_dir,
        current_run_id=run_id,
        topic_name=str(topic_state.get("name") or topic.get("name") or slug),
    )

    delta = build_monitor_delta(
        run_id=run_id,
        topic_name=str(topic_state.get("name") or topic.get("name") or slug),
        topic_state=topic_state,
        baseline_run_id=baseline_run_id,
        baseline_topic_state=baseline_topic_state,
    )

    monitor_errors: list[str] = []
    decisions: list[dict[str, Any]] = []

    if baseline_run_id is None:
        decisions.append(
            {
                "trigger_id": "monitor_no_baseline",
                "type": "system",
                "severity": "low",
                "fired": True,
                "reason": "No previous successful run baseline",
                "observed": {"baseline": None},
                "threshold": {},
                "topic": str(topic_state.get("name") or topic.get("name") or slug),
                "run_id": run_id,
                "emitted": True,
                "suppressed_by": None,
            }
        )
    else:
        r_decisions, r_errors = evaluate_monitor_rules(
            topic=str(topic_state.get("name") or topic.get("name") or slug),
            run_id=run_id,
            delta=delta,
            triggers=triggers,
        )
        decisions.extend(r_decisions)
        monitor_errors.extend(r_errors)

    if bool(((mon_global.get("hooks", {}) or {}).get("enabled", True))) and hooks:
        h_decisions, h_errors = run_monitor_hooks(
            hooks=hooks,
            allowlist=list((hooks_global.get("allowlist", []) or [])),
            timeout_sec=float(hooks_global.get("timeout_sec", 5)),
            context={
                "config": cfg,
                "topic": topic,
                "topic_state": topic_state,
                "delta": delta,
                "run_id": run_id,
                "baseline_run_id": baseline_run_id,
            },
            topic=str(topic_state.get("name") or topic.get("name") or slug),
            run_id=run_id,
        )
        decisions.extend(h_decisions)
        monitor_errors.extend(h_errors)

    state_path = Path(audit_dir) / "monitor_state.json"
    state = load_monitor_state(str(state_path))

    topic_mon = (topic.get("monitoring", {}) if isinstance(topic, dict) else {}) or {}
    cooldown = int(topic_mon.get("cooldown_minutes", noise_defaults.get("cooldown_minutes", 360)))
    hysteresis = int(topic_mon.get("hysteresis_runs", noise_defaults.get("hysteresis_runs", 2)))
    decisions = apply_noise_controls(
        topic=str(topic_state.get("name") or topic.get("name") or slug),
        decisions=decisions,
        state=state,
        cooldown_minutes=cooldown,
        hysteresis_runs=hysteresis,
        now=_utcnow(),
    )
    save_monitor_state(str(state_path), state)

    events: list[dict[str, Any]] = []
    digest_queue_path = Path(
        str((mon_global.get("digest", {}) or {}).get("path", "runs/audit/monitor_digest_queue.json"))
    )
    digest_enabled = bool((mon_global.get("digest", {}) or {}).get("enabled", True))

    for d in decisions:
        if not bool(d.get("fired", False)):
            continue
        if not bool(d.get("emitted", False)):
            continue
        sev = str(d.get("severity", "low")).lower()
        if sev == "high":
            events.append(
                {
                    "severity": "critical",
                    "code": "monitor_trigger_high",
                    "message": f"Monitoring trigger fired ({d.get('trigger_id')})",
                    "details": d,
                }
            )
        elif sev == "medium":
            item = {
                "queued_at": _utcnow().isoformat(),
                "run_id": run_id,
                "topic": d.get("topic"),
                "decision": d,
            }
            if digest_enabled:
                queue = []
                if digest_queue_path.exists():
                    try:
                        queue = json.loads(digest_queue_path.read_text(encoding="utf-8"))
                    except Exception:
                        queue = []
                queue.append(item)
                digest_queue_path.parent.mkdir(parents=True, exist_ok=True)
                digest_queue_path.write_text(json.dumps(queue, indent=2), encoding="utf-8")
            events.append(
                {
                    "severity": "warning",
                    "code": "monitor_trigger_medium_digest_queued",
                    "message": f"Monitoring trigger queued for digest ({d.get('trigger_id')})",
                    "details": d,
                }
            )
        else:
            # low severity is log-only (artifact visibility, no immediate alert routing)
            pass

    if monitor_errors:
        for err in monitor_errors:
            events.append(
                {
                    "severity": "critical",
                    "code": "monitoring_error",
                    "message": "Monitoring evaluation error (fail-open)",
                    "details": {
                        "topic": str(topic_state.get("name") or topic.get("name") or slug),
                        "error": err,
                    },
                }
            )

    run_dir_path = Path(run_dir)
    delta_path = _write_json(run_dir_path / f"monitor_{slug}.delta.json", delta)
    triggers_path = _write_json(
        run_dir_path / f"monitor_{slug}.triggers.json",
        {
            "run_id": run_id,
            "topic": str(topic_state.get("name") or topic.get("name") or slug),
            "baseline_run_id": baseline_run_id,
            "cooldown_minutes": cooldown,
            "hysteresis_runs": hysteresis,
            "decisions": decisions,
            "errors": monitor_errors,
        },
    )

    status = "ok"
    if monitor_errors:
        status = "error"
    elif any(e.get("code") == "monitor_trigger_high" for e in events):
        status = "warn"

    return {
        "monitoring_enabled": True,
        "monitor_baseline_run_id": baseline_run_id,
        "monitor_diff_path": delta_path,
        "monitor_trigger_results_path": triggers_path,
        "monitor_events": events,
        "monitor_events_count": len(events),
        "monitor_errors_count": len(monitor_errors),
        "monitor_errors": monitor_errors,
        "monitor_status": status,
    }


def monitor_digest_flush(*, config: dict[str, Any]) -> dict[str, Any]:
    mon = (config.get("monitoring", {}) if isinstance(config, dict) else {}) or {}
    digest = (mon.get("digest", {}) if isinstance(mon, dict) else {}) or {}
    path = Path(str(digest.get("path", "runs/audit/monitor_digest_queue.json")))
    if not path.exists():
        return {"queued": 0, "flushed": 0, "events": []}
    try:
        queue = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        queue = []
    if not isinstance(queue, list):
        queue = []

    grouped: dict[str, list[dict[str, Any]]] = {}
    for item in queue:
        topic = str((item or {}).get("topic", "unknown"))
        grouped.setdefault(topic, []).append(item)

    events = []
    for topic, items in grouped.items():
        events.append(
            {
                "severity": "warning",
                "code": "monitor_digest",
                "message": f"Monitoring digest for {topic}",
                "details": {"topic": topic, "count": len(items), "items": items},
            }
        )

    # Atomic clear
    path.write_text("[]\n", encoding="utf-8")
    return {"queued": len(queue), "flushed": len(queue), "events": events}


def monitor_status(*, config: dict[str, Any]) -> dict[str, Any]:
    mon = (config.get("monitoring", {}) if isinstance(config, dict) else {}) or {}
    digest = (mon.get("digest", {}) if isinstance(mon, dict) else {}) or {}
    path = Path(str(digest.get("path", "runs/audit/monitor_digest_queue.json")))
    queue_len = 0
    if path.exists():
        try:
            q = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(q, list):
                queue_len = len(q)
        except Exception:
            queue_len = 0

    audit_dir = str((config.get("paths", {}) or {}).get("audit_dir", "runs/audit"))
    state = load_monitor_state(str(Path(audit_dir) / "monitor_state.json"))
    schedule_dir = Path("runs/monitor/schedules")
    schedule_entries: list[dict[str, Any]] = []
    schedule_by_backend: dict[str, int] = {}
    if schedule_dir.exists():
        for path in sorted(schedule_dir.glob("*.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if not isinstance(payload, dict):
                continue
            backend = str(payload.get("backend", "unknown"))
            schedule_by_backend[backend] = int(schedule_by_backend.get(backend, 0)) + 1
            schedule_entries.append(
                {
                    "name": payload.get("name", path.stem),
                    "schedule": payload.get("schedule"),
                    "backend": backend,
                    "updated_utc": payload.get("updated_utc"),
                    "path": str(path),
                }
            )
    return {
        "monitoring_enabled_default": bool(mon.get("enabled_default", False)),
        "baseline": mon.get("baseline", "previous_successful_run"),
        "failure_policy": mon.get("failure_policy", "fail_open"),
        "digest_queue_path": str(path),
        "digest_queue_count": queue_len,
        "schedule_registry_dir": str(schedule_dir),
        "schedule_registry_count": len(schedule_entries),
        "schedule_registry_by_backend": schedule_by_backend,
        "schedule_registry_entries": schedule_entries,
        "state": state,
    }
