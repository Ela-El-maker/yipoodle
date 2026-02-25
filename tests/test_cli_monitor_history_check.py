import json

from src.cli import build_parser


def test_cli_monitor_history_check_writes_output(tmp_path) -> None:
    run_dir = tmp_path / "runs" / "20260224T150000Z"
    run_dir.mkdir(parents=True, exist_ok=True)
    triggers = {
        "run_id": "20260224T150000Z",
        "topic": "finance_risk",
        "cooldown_minutes": 360,
        "decisions": [
            {
                "trigger_id": "claim_shift",
                "severity": "high",
                "fired": True,
                "emitted": True,
                "suppressed_by": "none",
            }
        ],
    }
    (run_dir / "monitor_finance_risk.triggers.json").write_text(json.dumps(triggers, indent=2), encoding="utf-8")

    p = build_parser()
    out_path = tmp_path / "history.json"
    args = p.parse_args(
        [
            "monitor-history-check",
            "--topic",
            "finance_risk",
            "--audit-dir",
            str(tmp_path),
            "--out",
            str(out_path),
        ]
    )
    args.func(args)

    assert out_path.exists()
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["topic"] == "finance_risk"
    assert payload["runs_evaluated"] == 1
    assert payload["emitted_total"] == 1
