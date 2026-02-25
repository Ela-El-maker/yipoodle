from __future__ import annotations

from pathlib import Path

from src.apps.reliability_watchdog import run_reliability_watchdog


def test_reliability_watchdog_disabled() -> None:
    out = run_reliability_watchdog(sync_stats_list=[], reliability_cfg={"enabled": False}, run_id="r1")
    assert out["enabled"] is False
    assert out["events"] == []


def test_reliability_watchdog_records_and_emits(tmp_path) -> None:
    db = tmp_path / "reliability.db"
    state = tmp_path / "state.json"
    report = tmp_path / "report.json"
    cfg = {
        "enabled": True,
        "db_path": str(db),
        "state_path": str(state),
        "report_path": str(report),
        "degrade_threshold": 0.9,
        "critical_threshold": 0.4,
        "auto_disable_after": 1,
        "source_mapping": {"from_arxiv": "arxiv"},
    }
    sync_stats = [{"from_arxiv": 0, "source_errors": 2}]
    out = run_reliability_watchdog(sync_stats_list=sync_stats, reliability_cfg=cfg, run_id="r1")
    assert out["enabled"] is True
    assert Path(out["report_path"]).exists()
    assert state.exists()
    assert any(ev["code"] in {"source_reliability_degraded", "source_reliability_critical"} for ev in out["events"])


def test_reliability_watchdog_resets_below_runs_on_recovery(tmp_path) -> None:
    db = tmp_path / "reliability.db"
    state = tmp_path / "state.json"
    report = tmp_path / "report.json"
    cfg = {
        "enabled": True,
        "db_path": str(db),
        "state_path": str(state),
        "report_path": str(report),
        "degrade_threshold": 0.4,
        "critical_threshold": 0.2,
        "auto_disable_after": 3,
        "source_mapping": {"from_arxiv": "arxiv"},
    }
    _ = run_reliability_watchdog(
        sync_stats_list=[{"from_arxiv": 0, "source_errors": 1}],
        reliability_cfg=cfg,
        run_id="r1",
    )
    out = run_reliability_watchdog(
        sync_stats_list=[{"from_arxiv": 5, "source_errors": 0}],
        reliability_cfg=cfg,
        run_id="r2",
    )
    arxiv = next(x for x in out["sources"] if x["source"] == "arxiv")
    assert arxiv["below_runs"] == 0
