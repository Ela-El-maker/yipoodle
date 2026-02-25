from __future__ import annotations

import argparse
import json


def test_cli_parser_new_automation_commands() -> None:
    from src.cli import build_parser

    p = build_parser()
    args = p.parse_args(["run-automation", "--config", "config/automation.yaml"])
    assert args.command == "run-automation"

    args = p.parse_args(["research-template", "--template", "lit_review", "--topic", "x", "--index", "idx.json"])
    assert args.command == "research-template"

    args = p.parse_args(["watch-ingest", "--dir", "data/papers", "--once"])
    assert args.command == "watch-ingest"

    args = p.parse_args(["benchmark-regression-check", "--benchmark", "runs/b.json"])
    assert args.command == "benchmark-regression-check"

    args = p.parse_args(["reliability-watchdog", "--sync-stats", "runs/s1.json", "--out", "runs/out.json"])
    assert args.command == "reliability-watchdog"

    args = p.parse_args(["kb-contradiction-resolve", "--kb-db", "data/kb/knowledge.db", "--topic", "finance", "--index", "idx.json"])
    assert args.command == "kb-contradiction-resolve"


def test_cmd_benchmark_regression_check_fail_on_regression(monkeypatch, capsys) -> None:
    from src.cli import cmd_benchmark_regression_check

    monkeypatch.setattr(
        "src.cli.run_benchmark_regression_check",
        lambda **_kwargs: {"regressed": True, "reasons": ["x"]},
    )
    args = argparse.Namespace(
        benchmark="b.json",
        history="h.json",
        run_id="r1",
        max_latency_regression_pct=10.0,
        min_quality_floor=0.0,
        history_window=10,
        fail_on_regression=True,
        out=None,
    )
    try:
        cmd_benchmark_regression_check(args)
    except SystemExit as exc:
        assert "benchmark regression gate failed" in str(exc)
    else:
        raise AssertionError("expected SystemExit")
    assert "regressed" in capsys.readouterr().out


def test_cmd_run_automation_prints_run_dir(monkeypatch, capsys) -> None:
    from src.cli import cmd_run_automation

    monkeypatch.setattr("src.cli.run_automation", lambda _config: "runs/audit/runs/20260225T000000Z")
    args = argparse.Namespace(config="config/automation.yaml")
    cmd_run_automation(args)
    assert "runs/audit/runs/20260225T000000Z" in capsys.readouterr().out


def test_cmd_reliability_watchdog_from_sync_stats(monkeypatch, tmp_path, capsys) -> None:
    from src.cli import cmd_reliability_watchdog

    stats = tmp_path / "sync.json"
    stats.write_text(json.dumps({"from_arxiv": 2, "source_errors": 0}), encoding="utf-8")

    monkeypatch.setattr(
        "src.cli.run_reliability_watchdog",
        lambda **kwargs: {"enabled": True, "run_id": kwargs["run_id"], "events": [], "sources": []},
    )
    args = argparse.Namespace(
        config=None,
        run_dir=None,
        sync_stats=[str(stats)],
        run_id="r1",
        reliability_db=None,
        state_path=None,
        report_path=None,
        degrade_threshold=None,
        critical_threshold=None,
        auto_disable_after=None,
        out=None,
    )
    cmd_reliability_watchdog(args)
    out = capsys.readouterr().out
    assert '"enabled": true' in out


def test_cmd_reliability_watchdog_writes_out(monkeypatch, tmp_path) -> None:
    from src.cli import cmd_reliability_watchdog

    stats = tmp_path / "sync.json"
    stats.write_text(json.dumps({"from_arxiv": 1, "source_errors": 0}), encoding="utf-8")
    out_path = tmp_path / "watchdog.json"
    monkeypatch.setattr(
        "src.cli.run_reliability_watchdog",
        lambda **_kwargs: {"enabled": True, "run_id": "r2", "events": [], "sources": []},
    )
    args = argparse.Namespace(
        config=None,
        run_dir=None,
        sync_stats=[str(stats)],
        run_id="r2",
        reliability_db=None,
        state_path=None,
        report_path=None,
        degrade_threshold=None,
        critical_threshold=None,
        auto_disable_after=None,
        out=str(out_path),
    )
    cmd_reliability_watchdog(args)
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["enabled"] is True
