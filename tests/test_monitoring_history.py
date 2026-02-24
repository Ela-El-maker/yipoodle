import json
from pathlib import Path

from src.apps.monitoring_history import run_monitor_history_check


def _write_triggers(
    base: Path,
    *,
    run_id: str,
    topic_slug: str,
    cooldown_minutes: int,
    decisions: list[dict],
) -> None:
    run_dir = base / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_id": run_id,
        "topic": topic_slug,
        "cooldown_minutes": cooldown_minutes,
        "decisions": decisions,
    }
    (run_dir / f"monitor_{topic_slug}.triggers.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def test_monitor_history_check_aggregates_counts_and_writes_output(tmp_path) -> None:
    _write_triggers(
        tmp_path,
        run_id="20260224T150000Z",
        topic_slug="finance_risk",
        cooldown_minutes=360,
        decisions=[
            {
                "trigger_id": "claim_shift",
                "severity": "high",
                "fired": True,
                "emitted": True,
                "suppressed_by": "none",
            },
            {
                "trigger_id": "new_docs",
                "severity": "medium",
                "fired": False,
                "emitted": False,
                "suppressed_by": "not_fired",
            },
        ],
    )
    _write_triggers(
        tmp_path,
        run_id="20260224T210000Z",
        topic_slug="finance_risk",
        cooldown_minutes=360,
        decisions=[
            {
                "trigger_id": "claim_shift",
                "severity": "high",
                "fired": True,
                "emitted": False,
                "suppressed_by": "cooldown",
            }
        ],
    )
    out = tmp_path / "out.json"
    payload = run_monitor_history_check(topic="finance_risk", audit_dir=str(tmp_path), out_path=str(out))

    assert payload["runs_evaluated"] == 2
    assert payload["decisions_total"] == 3
    assert payload["fired_total"] == 2
    assert payload["emitted_total"] == 1
    assert payload["emitted_by_severity"]["high"] == 1
    assert payload["suppressed_counts"]["cooldown"] == 1
    assert out.exists()
    persisted = json.loads(out.read_text(encoding="utf-8"))
    assert persisted["topic"] == "finance_risk"


def test_monitor_history_check_detects_cooldown_violation(tmp_path) -> None:
    _write_triggers(
        tmp_path,
        run_id="20260224T150000Z",
        topic_slug="finance_risk",
        cooldown_minutes=360,
        decisions=[
            {
                "trigger_id": "claim_shift",
                "severity": "high",
                "fired": True,
                "emitted": True,
                "suppressed_by": "none",
            }
        ],
    )
    _write_triggers(
        tmp_path,
        run_id="20260224T160000Z",
        topic_slug="finance_risk",
        cooldown_minutes=360,
        decisions=[
            {
                "trigger_id": "claim_shift",
                "severity": "high",
                "fired": True,
                "emitted": True,
                "suppressed_by": "none",
            }
        ],
    )
    payload = run_monitor_history_check(topic="finance_risk", audit_dir=str(tmp_path))
    assert payload["cooldown_violation_count"] == 1
    viol = payload["cooldown_violations"][0]
    assert viol["trigger_id"] == "claim_shift"
    assert viol["delta_minutes"] == 60
    assert viol["required_cooldown_minutes"] == 360
