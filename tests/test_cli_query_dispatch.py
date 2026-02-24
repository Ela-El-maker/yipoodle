import json
from pathlib import Path

from src.cli import build_parser


def test_cli_query_dispatch_ask_writes_router_sidecar(tmp_path) -> None:
    p = build_parser()
    out = tmp_path / "ask.md"
    args = p.parse_args(["query", "--question", "23 + 34 = ?", "--mode", "ask", "--out", str(out)])
    args.func(args)

    assert out.exists()
    router = json.loads(out.with_suffix(".router.json").read_text(encoding="utf-8"))
    assert router["mode_selected"] == "ask"
    assert router["override_used"] is True


def test_cli_query_dispatch_monitor_auto(tmp_path, monkeypatch) -> None:
    p = build_parser()

    def _fake_monitor_mode(*, question, schedule, automation_config_path, out_path, register_schedule, schedule_backend):
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text("{}", encoding="utf-8")
        return {
            "mode": "monitor",
            "query": question,
            "schedule": schedule,
            "monitor_spec_path": "runs/monitor/topics/x.json",
            "generated_automation_config": "runs/monitor/generated/x.automation.yaml",
            "schedule_register_requested": register_schedule,
            "schedule_backend_requested": schedule_backend,
            "schedule_backend_used": "file",
            "schedule_registered": False,
            "schedule_entry": None,
            "schedule_error": None,
            "monitor_bootstrap_ok": True,
            "baseline_run_id": "20260224T150000Z",
            "baseline_run_dir": "runs/monitor/audit/x/runs/20260224T150000Z",
        }

    monkeypatch.setattr("src.cli.run_monitor_mode", _fake_monitor_mode)

    out = tmp_path / "monitor.md"
    args = p.parse_args(["query", "--question", "Monitor NVIDIA stock", "--out", str(out)])
    args.func(args)

    router = json.loads(out.with_suffix(".router.json").read_text(encoding="utf-8"))
    assert router["mode_selected"] == "monitor"
    assert out.exists()


def test_cli_query_dispatch_notes_auto(tmp_path, monkeypatch) -> None:
    p = build_parser()

    def _fake_notes_mode(*, question, index_path, kb_db, topic, out_path, sources_config_path=None):
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text("# Study Notes\n", encoding="utf-8")
        return {
            "mode": "notes",
            "notes_path": out_path,
            "kb_ingest_succeeded": True,
            "kb_claims_added": 1,
            "kb_claims_updated": 0,
        }

    monkeypatch.setattr("src.cli.run_notes_mode", _fake_notes_mode)

    out = tmp_path / "notes.md"
    args = p.parse_args(["query", "--question", "Create study notes on transformers", "--out", str(out)])
    args.func(args)

    router = json.loads(out.with_suffix(".router.json").read_text(encoding="utf-8"))
    assert router["mode_selected"] == "notes"
    assert out.exists()


def test_cli_query_dispatch_research_auto(tmp_path, monkeypatch) -> None:
    p = build_parser()

    def _fake_run_research(*_args, **kwargs):
        out = Path(_args[3])
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("# Research Report\n", encoding="utf-8")
        return str(out)

    monkeypatch.setattr("src.cli.run_research", _fake_run_research)

    out = tmp_path / "research.md"
    args = p.parse_args([
        "query",
        "--question",
        "Compare sparse transformers and RNNs for long-context forecasting",
        "--out",
        str(out),
    ])
    args.func(args)

    router = json.loads(out.with_suffix(".router.json").read_text(encoding="utf-8"))
    assert router["mode_selected"] == "research"
    assert out.exists()


def test_cli_parser_new_mode_commands() -> None:
    p = build_parser()
    q = p.parse_args(["query", "--question", "x"])
    assert q.command == "query"
    a = p.parse_args(["ask", "--question", "x"])
    assert a.command == "ask"
    m = p.parse_args(["monitor", "--question", "x"])
    assert m.command == "monitor"
    n = p.parse_args(["notes", "--question", "x"])
    assert n.command == "notes"


def test_cli_monitor_uses_router_default_schedule(tmp_path, monkeypatch) -> None:
    p = build_parser()
    captured: dict[str, str] = {}

    monkeypatch.setattr(
        "src.cli.load_router_config",
        lambda _path: {"router": {"monitor": {"default_schedule_cron": "0 */3 * * *"}}},
    )

    def _fake_monitor_mode(*, question, schedule, automation_config_path, out_path, register_schedule, schedule_backend):
        captured["schedule"] = schedule
        captured["register_schedule"] = str(register_schedule)
        captured["schedule_backend"] = str(schedule_backend)
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text("{}", encoding="utf-8")
        return {"mode": "monitor", "schedule": schedule}

    monkeypatch.setattr("src.cli.run_monitor_mode", _fake_monitor_mode)

    out = tmp_path / "monitor.json"
    args = p.parse_args(["monitor", "--question", "Monitor PIX", "--out", str(out)])
    args.func(args)

    assert captured["schedule"] == "0 */3 * * *"
    assert captured["register_schedule"] == "True"
    assert captured["schedule_backend"] == "auto"
