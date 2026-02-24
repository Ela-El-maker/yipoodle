from datetime import datetime, timedelta, timezone

from src.apps.monitoring_state import apply_noise_controls


def test_monitoring_state_hysteresis_and_cooldown() -> None:
    state = {"topics": {}}
    base = datetime(2026, 2, 24, 12, 0, 0, tzinfo=timezone.utc)

    d = [{"trigger_id": "a", "fired": True, "severity": "high", "observed": 1}]
    out1 = apply_noise_controls(
        topic="t",
        decisions=d,
        state=state,
        cooldown_minutes=360,
        hysteresis_runs=2,
        now=base,
    )
    assert out1[0]["emitted"] is False
    assert out1[0]["suppressed_by"] == "hysteresis"

    out2 = apply_noise_controls(
        topic="t",
        decisions=d,
        state=state,
        cooldown_minutes=360,
        hysteresis_runs=2,
        now=base + timedelta(minutes=1),
    )
    assert out2[0]["emitted"] is True

    out3 = apply_noise_controls(
        topic="t",
        decisions=d,
        state=state,
        cooldown_minutes=360,
        hysteresis_runs=1,
        now=base + timedelta(minutes=2),
    )
    assert out3[0]["emitted"] is False
    assert out3[0]["suppressed_by"] == "cooldown"
