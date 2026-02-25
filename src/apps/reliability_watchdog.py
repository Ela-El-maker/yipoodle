from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import json
from typing import Any

from src.apps.source_reliability import DEFAULT_RELIABILITY_DB, record_feedback, recompute_reliability

SYNC_SOURCE_FIELD_MAP = {
    "from_arxiv": "arxiv",
    "from_openalex": "openalex",
    "from_semanticscholar": "semanticscholar",
    "from_crossref": "crossref",
    "from_dblp": "dblp",
    "from_paperswithcode": "paperswithcode",
    "from_core": "core",
    "from_openreview": "openreview",
    "from_github": "github",
    "from_zenodo": "zenodo",
    "from_opencitations": "opencitations",
    "from_springer": "springer",
    "from_ieee_xplore": "ieee_xplore",
    "from_figshare": "figshare",
    "from_openml": "openml",
    "from_gdelt": "gdelt",
    "from_wikidata": "wikidata",
    "from_orcid": "orcid",
}


def run_reliability_watchdog(
    *,
    sync_stats_list: list[dict[str, Any]],
    reliability_cfg: dict[str, Any] | None = None,
    run_id: str,
) -> dict[str, Any]:
    cfg = reliability_cfg or {}
    enabled = bool(cfg.get("enabled", False))
    if not enabled:
        return {"enabled": False, "events": [], "sources": []}

    db_path = str(cfg.get("db_path", DEFAULT_RELIABILITY_DB))
    state_path = Path(str(cfg.get("state_path", "runs/audit/source_reliability_state.json")))
    report_path = Path(str(cfg.get("report_path", "runs/audit/source_reliability.json")))
    degrade_threshold = float(cfg.get("degrade_threshold", 0.30))
    critical_threshold = float(cfg.get("critical_threshold", 0.15))
    auto_disable_after = int(cfg.get("auto_disable_after", 0) or 0)
    custom_map = cfg.get("source_mapping", {}) or {}
    source_map = dict(SYNC_SOURCE_FIELD_MAP)
    if isinstance(custom_map, dict):
        for k, v in custom_map.items():
            source_map[str(k)] = str(v)

    by_source_totals: dict[str, int] = {}
    total_source_errors = 0
    touched_sources: set[str] = set()
    for stats in sync_stats_list:
        total_source_errors += int(stats.get("source_errors", 0) or 0)
        for field, source in source_map.items():
            c = int(stats.get(field, 0) or 0)
            by_source_totals[source] = int(by_source_totals.get(source, 0)) + c

    for source, cnt in by_source_totals.items():
        if cnt > 0:
            record_feedback(db_path, source, "fetch_success", value=float(cnt), run_id=run_id)
            touched_sources.add(source)

    # sync-papers currently exposes aggregate source_errors; spread across
    # zero-result sources to keep feedback loop informative.
    if total_source_errors > 0:
        zero_sources = [name for name, cnt in by_source_totals.items() if cnt <= 0]
        if zero_sources:
            per_source_err = max(1.0, float(total_source_errors) / float(len(zero_sources)))
            for source in zero_sources:
                record_feedback(db_path, source, "fetch_error", value=per_source_err, run_id=run_id)
                touched_sources.add(source)

    state: dict[str, Any] = {}
    if state_path.exists():
        try:
            payload = json.loads(state_path.read_text(encoding="utf-8"))
            state = payload if isinstance(payload, dict) else {}
        except Exception:
            state = {}

    scores: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    for source in sorted(touched_sources):
        score = float(recompute_reliability(db_path, source))
        prev = state.get(source, {}) if isinstance(state.get(source), dict) else {}
        below_runs = int(prev.get("below_runs", 0) or 0)
        if score < degrade_threshold:
            below_runs += 1
        else:
            below_runs = 0
        auto_disable_recommended = bool(auto_disable_after > 0 and below_runs >= auto_disable_after)
        state[source] = {
            "score": round(score, 4),
            "below_runs": below_runs,
            "last_run_id": run_id,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "auto_disable_recommended": auto_disable_recommended,
        }
        if score < critical_threshold:
            events.append(
                {
                    "severity": "critical",
                    "code": "source_reliability_critical",
                    "message": f"Source reliability below critical threshold: {source}",
                    "details": {
                        "source": source,
                        "score": round(score, 4),
                        "critical_threshold": critical_threshold,
                        "below_runs": below_runs,
                        "auto_disable_recommended": auto_disable_recommended,
                    },
                }
            )
        elif score < degrade_threshold:
            events.append(
                {
                    "severity": "warning",
                    "code": "source_reliability_degraded",
                    "message": f"Source reliability below degrade threshold: {source}",
                    "details": {
                        "source": source,
                        "score": round(score, 4),
                        "degrade_threshold": degrade_threshold,
                        "below_runs": below_runs,
                        "auto_disable_recommended": auto_disable_recommended,
                    },
                }
            )
        scores.append(
            {
                "source": source,
                "score": round(score, 4),
                "fetched_items": int(by_source_totals.get(source, 0)),
                "below_runs": below_runs,
                "auto_disable_recommended": auto_disable_recommended,
            }
        )

    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
    payload = {
        "enabled": True,
        "run_id": run_id,
        "db_path": db_path,
        "degrade_threshold": degrade_threshold,
        "critical_threshold": critical_threshold,
        "auto_disable_after": auto_disable_after,
        "aggregate_source_errors": total_source_errors,
        "sources": scores,
        "events": events,
        "state_path": str(state_path),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    payload["report_path"] = str(report_path)
    return payload
