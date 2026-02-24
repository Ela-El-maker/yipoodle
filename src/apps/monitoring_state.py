from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
import json
from typing import Any


def load_monitor_state(path: str) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {"topics": {}}
    try:
        payload = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {"topics": {}}
    if not isinstance(payload, dict):
        return {"topics": {}}
    payload.setdefault("topics", {})
    return payload


def save_monitor_state(path: str, state: dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(state, indent=2), encoding="utf-8")


def _topic_trigger_state(state: dict[str, Any], topic: str, trigger_id: str) -> dict[str, Any]:
    topics = state.setdefault("topics", {})
    t = topics.setdefault(topic, {})
    tr = t.setdefault("triggers", {})
    return tr.setdefault(trigger_id, {"last_fired_at": None, "consecutive_hits": 0, "last_value": None})


def apply_noise_controls(
    *,
    topic: str,
    decisions: list[dict[str, Any]],
    state: dict[str, Any],
    cooldown_minutes: int,
    hysteresis_runs: int,
    now: datetime | None = None,
) -> list[dict[str, Any]]:
    ts = now or datetime.now(timezone.utc)
    out: list[dict[str, Any]] = []
    for d in decisions:
        trigger_id = str(d.get("trigger_id") or "unnamed_trigger")
        tr_state = _topic_trigger_state(state, topic, trigger_id)
        fired = bool(d.get("fired", False))
        tr_state["last_value"] = d.get("observed")
        if not fired:
            tr_state["consecutive_hits"] = 0
            d["emitted"] = False
            d["suppressed_by"] = "not_fired"
            out.append(d)
            continue

        tr_state["consecutive_hits"] = int(tr_state.get("consecutive_hits", 0)) + 1

        if int(tr_state["consecutive_hits"]) < max(1, int(hysteresis_runs)):
            d["emitted"] = False
            d["suppressed_by"] = "hysteresis"
            out.append(d)
            continue

        last_fired_raw = tr_state.get("last_fired_at")
        if last_fired_raw:
            try:
                last_fired = datetime.fromisoformat(str(last_fired_raw))
                if ts - last_fired < timedelta(minutes=max(0, int(cooldown_minutes))):
                    d["emitted"] = False
                    d["suppressed_by"] = "cooldown"
                    out.append(d)
                    continue
            except Exception:
                pass

        tr_state["last_fired_at"] = ts.isoformat()
        tr_state["consecutive_hits"] = 0
        d["emitted"] = True
        d["suppressed_by"] = None
        out.append(d)
    return out
