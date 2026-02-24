from src.cli import build_parser


def test_cli_parser_monitoring_commands() -> None:
    p = build_parser()

    args = p.parse_args(["monitor-status", "--config", "config/automation.yaml"])
    assert args.command == "monitor-status"

    args = p.parse_args(["monitor-digest-flush", "--config", "config/automation.yaml"])
    assert args.command == "monitor-digest-flush"

    args = p.parse_args(
        [
            "monitor-evaluate",
            "--config",
            "config/automation.yaml",
            "--run-dir",
            "runs/audit/runs/20260224T000000Z",
            "--topic",
            "finance",
        ]
    )
    assert args.command == "monitor-evaluate"
    assert args.topic == "finance"

    args = p.parse_args(["monitor-unregister", "--name", "finance_risk"])
    assert args.command == "monitor-unregister"
    assert args.name == "finance_risk"

    args = p.parse_args(["monitor-soak-sim", "--topic", "finance", "--runs", "48"])
    assert args.command == "monitor-soak-sim"
    assert args.topic == "finance"
    assert args.runs == 48

    args = p.parse_args(["monitor-history-check", "--topic", "finance"])
    assert args.command == "monitor-history-check"
    assert args.topic == "finance"

    args = p.parse_args(["monitor", "--question", "Monitor finance", "--schedule-backend", "file"])
    assert args.command == "monitor"
    assert args.schedule_backend == "file"

    args = p.parse_args(["query-router-eval", "--cases", "tests/fixtures/router_eval_cases.json"])
    assert args.command == "query-router-eval"
    args = p.parse_args(
        [
            "query-router-eval",
            "--cases",
            "tests/fixtures/router_eval_cases.json",
            "--strict-min-accuracy",
            "0.95",
        ]
    )
    assert args.command == "query-router-eval"
    assert args.strict_min_accuracy == 0.95
