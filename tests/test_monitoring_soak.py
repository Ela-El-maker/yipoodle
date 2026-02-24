from src.apps.monitoring_soak import run_monitor_soak_sim


def test_monitor_soak_constant_bad_emits_with_cooldown_and_hysteresis() -> None:
    out = run_monitor_soak_sim(
        topic="t",
        runs=24,
        interval_minutes=60,
        cooldown_minutes=360,
        hysteresis_runs=2,
        pattern="constant_bad",
        out_path=None,
    )
    assert out["fired_count"] == 24
    # Expected emissions: run indexes 1,7,13,19 under these settings.
    assert out["emitted_count"] == 4
    assert out["suppressed_counts"]["hysteresis"] >= 1
    assert out["suppressed_counts"]["cooldown"] >= 1


def test_monitor_soak_pulse_pattern_can_prevent_emission() -> None:
    out = run_monitor_soak_sim(
        topic="t",
        runs=20,
        interval_minutes=60,
        cooldown_minutes=360,
        hysteresis_runs=2,
        pattern="pulse",
        trigger_every=4,
        out_path=None,
    )
    # Pulses are not consecutive, so hysteresis should suppress all.
    assert out["emitted_count"] == 0
    assert out["suppressed_counts"]["hysteresis"] > 0


def test_monitor_soak_burst_pattern_emits() -> None:
    out = run_monitor_soak_sim(
        topic="t",
        runs=24,
        interval_minutes=60,
        cooldown_minutes=360,
        hysteresis_runs=2,
        pattern="burst",
        burst_len=2,
        gap_len=4,
        out_path=None,
    )
    assert out["emitted_count"] >= 3
    assert len(out["timeline"]) == 24
