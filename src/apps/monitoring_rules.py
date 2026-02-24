from __future__ import annotations

from typing import Any


def _num(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return float(default)


def _cmp(lhs: float, op: str, rhs: float) -> bool:
    if op == "lt":
        return lhs < rhs
    if op == "lte":
        return lhs <= rhs
    if op == "gt":
        return lhs > rhs
    if op == "gte":
        return lhs >= rhs
    if op == "eq":
        return lhs == rhs
    if op == "neq":
        return lhs != rhs
    return False


def evaluate_monitor_rules(
    *,
    topic: str,
    run_id: str,
    delta: dict[str, Any],
    triggers: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    decisions: list[dict[str, Any]] = []
    errors: list[str] = []

    for trig in triggers or []:
        trig_id = str(trig.get("id") or trig.get("trigger_id") or "unnamed_trigger")
        trig_type = str(trig.get("type") or "").strip()
        severity = str(trig.get("severity") or "low").lower()
        reason = ""
        fired = False
        observed: dict[str, Any] = {}
        threshold: dict[str, Any] = {}

        try:
            if trig_type == "new_documents":
                min_new = int(trig.get("min_new_sources", 1))
                observed = {"new_sources_count": int(delta.get("new_sources_count", 0))}
                threshold = {"min_new_sources": min_new}
                fired = int(observed["new_sources_count"]) >= min_new
                reason = f"new_sources_count={observed['new_sources_count']} threshold={min_new}"

            elif trig_type == "kb_diff_count":
                counts = (delta.get("kb_diff_counts", {}) or {})
                min_added = int(trig.get("min_added_claims", 0))
                min_updated = int(trig.get("min_updated_claims", 0))
                min_disputed = int(trig.get("min_disputed_claims", 0))
                observed = {
                    "added": int(counts.get("added", 0)),
                    "updated": int(counts.get("updated", 0)),
                    "disputed": int(counts.get("disputed", 0)),
                }
                threshold = {
                    "min_added_claims": min_added,
                    "min_updated_claims": min_updated,
                    "min_disputed_claims": min_disputed,
                }
                fired = (
                    observed["added"] >= min_added > 0
                    or observed["updated"] >= min_updated > 0
                    or observed["disputed"] >= min_disputed > 0
                )
                reason = (
                    f"kb_counts added={observed['added']} updated={observed['updated']} "
                    f"disputed={observed['disputed']}"
                )

            elif trig_type == "metric_threshold":
                metric = str(trig.get("metric") or "")
                op = str(trig.get("op") or "lt")
                value = _num(trig.get("value"), 0.0)
                cur = _num((delta.get("metrics_current", {}) or {}).get(metric), 0.0)
                observed = {"metric": metric, "value": cur}
                threshold = {"op": op, "value": value}
                fired = _cmp(cur, op, value)
                reason = f"metric {metric}={cur} {op} {value}"

            elif trig_type == "keyword_presence":
                target = str(trig.get("target") or "synthesis")
                hay = str((delta.get("report_text", {}) or {}).get(target, ""))
                keywords = [str(x).strip() for x in (trig.get("keywords") or []) if str(x).strip()]
                hits = [k for k in keywords if k.lower() in hay.lower()]
                observed = {"target": target, "hits": hits}
                threshold = {"keywords": keywords}
                fired = len(hits) > 0
                reason = f"keyword hits={hits}" if hits else "no keyword hits"

            else:
                errors.append(f"unknown trigger type '{trig_type}' for trigger '{trig_id}'")
                continue

            decisions.append(
                {
                    "trigger_id": trig_id,
                    "type": trig_type,
                    "severity": severity,
                    "fired": bool(fired),
                    "reason": reason,
                    "observed": observed,
                    "threshold": threshold,
                    "topic": topic,
                    "run_id": run_id,
                }
            )
        except Exception as exc:
            errors.append(f"trigger '{trig_id}' failed: {exc}")

    return decisions, errors
