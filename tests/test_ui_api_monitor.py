from __future__ import annotations

import time
from pathlib import Path

from fastapi.testclient import TestClient

import src.ui.executors as ui_exec
from src.ui.app import create_app


def _make_ui_config(tmp_path: Path) -> Path:
    cfg = tmp_path / "ui.yaml"
    cfg.write_text(
        "\n".join(
            [
                "max_workers: 1",
                f"run_db_path: {tmp_path / 'ui_runs.db'}",
                "artifacts_roots:",
                f"  - {tmp_path}",
                "  - runs/monitor",
                "  - runs/audit/runs",
                "job_timeout_sec:",
                "  ask: 30",
                "  research: 30",
                "  notes: 30",
                "  monitor: 30",
                "  query: 30",
                "  automation: 30",
            ]
        ),
        encoding="utf-8",
    )
    return cfg


def _wait_final(client: TestClient, run_id: str) -> dict:
    t0 = time.time()
    while True:
        body = client.get(f"/api/v1/runs/{run_id}").json()
        if body["status"] in {"done", "failed", "cancelled"}:
            return body
        if time.time() - t0 > 5:
            return body
        time.sleep(0.05)


def test_monitor_create_endpoint_enqueues_run(tmp_path: Path, monkeypatch) -> None:
    def _fake_monitor_mode(*, question, schedule, automation_config_path, out_path, register_schedule, schedule_backend):
        p = Path(out_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("{}", encoding="utf-8")
        return {
            "mode": "monitor",
            "name": "demo_monitor",
            "question": question,
            "query": question,
            "schedule": schedule,
            "monitor_spec_path": str(tmp_path / "runs" / "monitor" / "topics" / "demo_monitor.json"),
            "generated_automation_config": str(tmp_path / "runs" / "monitor" / "generated" / "demo_monitor.automation.yaml"),
            "monitor_bootstrap_ok": True,
            "baseline_run_id": "20260225T000000Z",
            "baseline_run_dir": str(tmp_path / "runs" / "audit" / "runs" / "20260225T000000Z"),
            "schedule_register_requested": register_schedule,
            "schedule_backend_requested": schedule_backend,
            "schedule_backend_used": "file",
            "schedule_registered": True,
            "schedule_entry": "entry",
            "schedule_error": None,
            "monitor_bootstrap_error": None,
        }

    monkeypatch.setattr(ui_exec, "run_monitor_mode", _fake_monitor_mode)

    cfg = _make_ui_config(tmp_path)
    app = create_app(str(cfg))
    with TestClient(app) as client:
        resp = client.post("/api/v1/monitor/topics", json={"question": "Monitor PIX outages"})
        assert resp.status_code == 200
        run_id = resp.json()["run_id"]

        final = _wait_final(client, run_id)
        assert final["status"] == "done"
        assert final["details"]["mode"] == "monitor"


def test_monitor_unregister_endpoint(tmp_path: Path, monkeypatch) -> None:
    from src.ui.routes import monitor as monitor_routes

    monkeypatch.setattr(
        monitor_routes,
        "unregister_monitor",
        lambda name_or_question, delete_files=False: {
            "mode": "monitor_unreg",
            "name": name_or_question,
            "schedule_removed": True,
            "delete_files_requested": delete_files,
        },
    )

    cfg = _make_ui_config(tmp_path)
    app = create_app(str(cfg))
    with TestClient(app) as client:
        resp = client.post("/api/v1/monitor/topics/demo/unregister")
        assert resp.status_code == 200
        body = resp.json()
        assert body["mode"] == "monitor_unreg"
        assert body["name"] == "demo"
