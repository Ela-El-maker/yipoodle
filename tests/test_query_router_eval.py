import json

import pytest

from src.cli import build_parser
from src.apps.query_router_eval import run_query_router_eval


def test_query_router_eval_fixture_has_nonzero_accuracy(tmp_path) -> None:
    out = tmp_path / "router_eval.json"
    payload = run_query_router_eval(
        cases_path="tests/fixtures/router_eval_cases.json",
        router_config_path="config/router.yaml",
        out_path=str(out),
    )
    assert payload["total"] >= 8
    assert 0.0 <= payload["accuracy"] <= 1.0
    assert out.exists()
    written = json.loads(out.read_text(encoding="utf-8"))
    assert written["total"] == payload["total"]
    assert written["gate_applied"] is False
    assert written["gate_passed"] is True


def test_query_router_eval_invalid_cases_raises(tmp_path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps([{"question": "x", "expected_mode": "invalid"}]), encoding="utf-8")
    with pytest.raises(ValueError):
        run_query_router_eval(cases_path=str(bad), router_config_path="config/router.yaml", out_path=None)


def test_query_router_eval_strict_gate_fields(tmp_path) -> None:
    cases = tmp_path / "cases.json"
    # Deliberate mismatch: this question routes to ASK, expected is RESEARCH.
    cases.write_text(json.dumps([{"question": "23 + 34 = ?", "expected_mode": "research"}]), encoding="utf-8")
    payload = run_query_router_eval(
        cases_path=str(cases),
        router_config_path="config/router.yaml",
        out_path=None,
        strict_min_accuracy=1.0,
    )
    assert payload["gate_applied"] is True
    assert payload["gate_passed"] is False
    assert payload["accuracy"] == 0.0


def test_query_router_eval_invalid_strict_threshold_raises() -> None:
    with pytest.raises(ValueError):
        run_query_router_eval(
            cases_path="tests/fixtures/router_eval_cases.json",
            router_config_path="config/router.yaml",
            out_path=None,
            strict_min_accuracy=1.5,
        )


def test_query_router_eval_cli_command(tmp_path) -> None:
    p = build_parser()
    out = tmp_path / "router_eval.json"
    args = p.parse_args(
        [
            "query-router-eval",
            "--cases",
            "tests/fixtures/router_eval_cases.json",
            "--config",
            "config/router.yaml",
            "--out",
            str(out),
        ]
    )
    args.func(args)
    assert out.exists()


def test_query_router_eval_cli_gate_fail(tmp_path) -> None:
    p = build_parser()
    cases = tmp_path / "cases.json"
    cases.write_text(json.dumps([{"question": "23 + 34 = ?", "expected_mode": "research"}]), encoding="utf-8")
    out = tmp_path / "router_eval_fail.json"
    args = p.parse_args(
        [
            "query-router-eval",
            "--cases",
            str(cases),
            "--config",
            "config/router.yaml",
            "--out",
            str(out),
            "--strict-min-accuracy",
            "1.0",
        ]
    )
    with pytest.raises(SystemExit):
        args.func(args)
    assert out.exists()
