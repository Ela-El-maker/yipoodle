from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
import json

from .monitoring_rules import evaluate_monitor_rules
from .monitoring_state import apply_noise_controls


def _run_id(ts: datetime) -> str:
    return ts.strftime("%Y%m%dT%H%M%SZ")


def _value_for_run(
    *,
    i: int,
    pattern: str,
    trigger_every: int,
    burst_len: int,
    gap_len: int,
    bad_value: float,
    good_value: float,
) -> float:
    p = (pattern or "constant_bad").strip().lower()
    if p == "constant_bad":
        return bad_value
    if p == "pulse":
        if trigger_every <= 0:
            trigger_every = 1
        return bad_value if (i % trigger_every == 0) else good_value
    if p == "burst":
        cycle = max(1, int(burst_len)) + max(0, int(gap_len))
        pos = i % cycle
        return bad_value if pos < max(1, int(burst_len)) else good_value
    raise ValueError(f"unknown soak pattern: {pattern}")


def run_monitor_soak_sim(
    *,
    topic: str,
    runs: int = 72,
    interval_minutes: int = 60,
    cooldown_minutes: int = 360,
    hysteresis_runs: int = 2,
    pattern: str = "constant_bad",
    trigger_every: int = 4,
    burst_len: int = 2,
    gap_len: int = 4,
    severity: str = "high",
    metric: str = "evidence_usage",
    op: str = "lt",
    threshold: float = 0.6,
    bad_value: float = 0.2,
    good_value: float = 0.9,
    out_path: str | None = None,
) -> dict[str, Any]:
    if runs <= 0:
        raise ValueError("runs must be > 0")
    if interval_minutes <= 0:
        raise ValueError("interval_minutes must be > 0")

    state: dict[str, Any] = {"topics": {}}
    start = datetime.now(timezone.utc)

    trigger = {
        "id": "soak_metric_trigger",
        "type": "metric_threshold",
        "severity": str(severity).lower(),
        "metric": metric,
        "op": op,
        "value": float(threshold),
    }

    timeline: list[dict[str, Any]] = []
    fired_count = 0
    emitted_count = 0
    suppressed_by: dict[str, int] = {"not_fired": 0, "hysteresis": 0, "cooldown": 0, "none": 0}

    for i in range(int(runs)):
        ts = start + timedelta(minutes=int(interval_minutes) * i)
        val = _value_for_run(
            i=i,
            pattern=pattern,
            trigger_every=int(trigger_every),
            burst_len=int(burst_len),
            gap_len=int(gap_len),
            bad_value=float(bad_value),
            good_value=float(good_value),
        )
        delta = {"metrics_current": {metric: val}}
        decisions, errors = evaluate_monitor_rules(
            topic=topic,
            run_id=_run_id(ts),
            delta=delta,
            triggers=[trigger],
        )
        if errors:
            raise RuntimeError("soak rule evaluation failed: " + "; ".join(errors))

        controlled = apply_noise_controls(
            topic=topic,
            decisions=decisions,
            state=state,
            cooldown_minutes=int(cooldown_minutes),
            hysteresis_runs=int(hysteresis_runs),
            now=ts,
        )
        d = controlled[0]
        fired = bool(d.get("fired", False))
        emitted = bool(d.get("emitted", False))
        sup = str(d.get("suppressed_by", "none") or "none")

        if fired:
            fired_count += 1
        if emitted:
            emitted_count += 1

        suppressed_by[sup if sup in suppressed_by else "none"] = suppressed_by.get(sup, 0) + 1

        timeline.append(
            {
                "index": i,
                "run_id": _run_id(ts),
                "ts": ts.isoformat(),
                "metric_value": val,
                "fired": fired,
                "emitted": emitted,
                "suppressed_by": sup,
            }
        )

    payload: dict[str, Any] = {
        "topic": topic,
        "runs": int(runs),
        "interval_minutes": int(interval_minutes),
        "cooldown_minutes": int(cooldown_minutes),
        "hysteresis_runs": int(hysteresis_runs),
        "pattern": pattern,
        "trigger_every": int(trigger_every),
        "burst_len": int(burst_len),
        "gap_len": int(gap_len),
        "severity": str(severity).lower(),
        "metric": metric,
        "op": op,
        "threshold": float(threshold),
        "bad_value": float(bad_value),
        "good_value": float(good_value),
        "fired_count": fired_count,
        "emitted_count": emitted_count,
        "suppressed_counts": suppressed_by,
        "timeline": timeline,
    }

    if out_path:
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    return payload
