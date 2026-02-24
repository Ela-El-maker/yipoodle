from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any
import json
import re


_RUN_ID_RE = re.compile(r"^\d{8}T\d{6}Z$")


def _slug(text: str) -> str:
    out = "".join(ch.lower() if ch.isalnum() else "_" for ch in text.strip())
    out = "_".join(part for part in out.split("_") if part)
    return out or "topic"


def _parse_run_ts(run_id: str) -> datetime | None:
    try:
        return datetime.strptime(run_id, "%Y%m%dT%H%M%SZ")
    except Exception:
        return None


def run_monitor_history_check(
    *,
    topic: str,
    audit_dir: str = "runs/audit",
    out_path: str | None = None,
) -> dict[str, Any]:
    slug = _slug(topic)
    root = Path(audit_dir) / "runs"
    files = sorted(root.glob(f"*/monitor_{slug}.triggers.json"))

    runs_evaluated = 0
    decisions_total = 0
    fired_total = 0
    emitted_total = 0
    emitted_by_severity: dict[str, int] = {"high": 0, "medium": 0, "low": 0}
    suppressed_counts: dict[str, int] = {"not_fired": 0, "hysteresis": 0, "cooldown": 0, "none": 0}
    trigger_stats: dict[str, dict[str, Any]] = {}

    emitted_times_by_trigger: dict[str, list[tuple[str, datetime, int]]] = {}
    cooldown_violations: list[dict[str, Any]] = []

    first_run_id = None
    last_run_id = None

    for f in files:
        try:
            payload = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        run_id = str(payload.get("run_id") or f.parent.name)
        if not _RUN_ID_RE.match(run_id):
            run_id = f.parent.name
        if first_run_id is None:
            first_run_id = run_id
        last_run_id = run_id

        cooldown = int(payload.get("cooldown_minutes", 0) or 0)
        decisions = list(payload.get("decisions", []) or [])
        runs_evaluated += 1
        decisions_total += len(decisions)

        for d in decisions:
            trig = str(d.get("trigger_id") or "unknown")
            sev = str(d.get("severity") or "low").lower()
            if sev not in emitted_by_severity:
                emitted_by_severity[sev] = 0
            st = trigger_stats.setdefault(
                trig,
                {
                    "trigger_id": trig,
                    "fired_count": 0,
                    "emitted_count": 0,
                    "suppressed_counts": {"not_fired": 0, "hysteresis": 0, "cooldown": 0, "none": 0},
                    "min_interval_minutes": None,
                },
            )

            fired = bool(d.get("fired", False))
            emitted = bool(d.get("emitted", False))
            sup = str(d.get("suppressed_by") or "none")

            if fired:
                fired_total += 1
                st["fired_count"] = int(st["fired_count"]) + 1

            if emitted:
                emitted_total += 1
                emitted_by_severity[sev] = int(emitted_by_severity.get(sev, 0)) + 1
                st["emitted_count"] = int(st["emitted_count"]) + 1
                ts = _parse_run_ts(run_id)
                if ts is not None:
                    emitted_times_by_trigger.setdefault(trig, []).append((run_id, ts, cooldown))

            if sup not in suppressed_counts:
                suppressed_counts[sup] = 0
            suppressed_counts[sup] = int(suppressed_counts.get(sup, 0)) + 1

            tr_s = st["suppressed_counts"]
            if sup not in tr_s:
                tr_s[sup] = 0
            tr_s[sup] = int(tr_s.get(sup, 0)) + 1

    for trig, pts in emitted_times_by_trigger.items():
        pts_sorted = sorted(pts, key=lambda x: x[1])
        prev_run = None
        prev_ts = None
        prev_cd = None
        min_delta = None
        for run_id, ts, cooldown in pts_sorted:
            if prev_ts is not None:
                delta_min = int((ts - prev_ts).total_seconds() // 60)
                if min_delta is None or delta_min < min_delta:
                    min_delta = delta_min
                req_cd = int(prev_cd or 0)
                if req_cd > 0 and delta_min < req_cd:
                    cooldown_violations.append(
                        {
                            "trigger_id": trig,
                            "prev_run_id": prev_run,
                            "run_id": run_id,
                            "delta_minutes": delta_min,
                            "required_cooldown_minutes": req_cd,
                        }
                    )
            prev_run = run_id
            prev_ts = ts
            prev_cd = cooldown
        if trig in trigger_stats:
            trigger_stats[trig]["min_interval_minutes"] = min_delta

    payload: dict[str, Any] = {
        "topic": topic,
        "slug": slug,
        "audit_dir": str(audit_dir),
        "files_considered": len(files),
        "runs_evaluated": runs_evaluated,
        "first_run_id": first_run_id,
        "last_run_id": last_run_id,
        "decisions_total": decisions_total,
        "fired_total": fired_total,
        "emitted_total": emitted_total,
        "emitted_by_severity": emitted_by_severity,
        "suppressed_counts": suppressed_counts,
        "trigger_stats": sorted(trigger_stats.values(), key=lambda x: x["trigger_id"]),
        "cooldown_violations": cooldown_violations,
        "cooldown_violation_count": len(cooldown_violations),
    }

    if out_path:
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    return payload
