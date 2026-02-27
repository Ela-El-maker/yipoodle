from __future__ import annotations

import time
from pathlib import Path

from fastapi.testclient import TestClient

from src.ui.app import create_app


def _make_ui_config(tmp_path: Path) -> Path:
    cfg = tmp_path / "ui.yaml"
    cfg.write_text(
        "\n".join(
            [
                "host: 127.0.0.1",
                "port: 8080",
                "max_workers: 1",
                f"run_db_path: {tmp_path / 'ui_runs.db'}",
                "artifacts_roots:",
                f"  - {tmp_path}",
                "  - runs/query",
                "  - runs/research_reports",
                "  - runs/monitor",
                "  - runs/notes",
                "  - runs/audit/runs",
                "sse_keepalive_sec: 1",
                "job_timeout_sec:",
                "  ask: 30",
                "  research: 30",
                "  notes: 30",
                "  monitor: 30",
                "  query: 30",
                "  automation: 30",
                "router_config: config/router.yaml",
                "sources_config: config/sources.yaml",
                "automation_config: config/automation.yaml",
                "kb_db: data/kb/knowledge.db",
            ]
        ),
        encoding="utf-8",
    )
    return cfg


def _wait_final(client: TestClient, run_id: str, timeout: float = 5.0) -> dict:
    t0 = time.time()
    while True:
        resp = client.get(f"/api/v1/runs/{run_id}")
        assert resp.status_code == 200
        body = resp.json()
        if body["status"] in {"done", "failed", "cancelled"}:
            return body
        if time.time() - t0 > timeout:
            return body
        time.sleep(0.05)


def test_api_create_list_get_run(tmp_path: Path) -> None:
    cfg = _make_ui_config(tmp_path)
    app = create_app(str(cfg))
    with TestClient(app) as client:
        resp = client.post(
            "/api/v1/runs",
            json={
                "mode": "ask",
                "question": "23 + 34 = ?",
                "sources_config": "config/sources.yaml",
                "automation_config": "config/automation.yaml",
                "options": {},
            },
        )
        assert resp.status_code == 200
        created = resp.json()
        run_id = created["run_id"]

        final = _wait_final(client, run_id)
        assert final["status"] == "done"

        lst = client.get("/api/v1/runs")
        assert lst.status_code == 200
        assert lst.json()["count"] >= 1

        one = client.get(f"/api/v1/runs/{run_id}")
        assert one.status_code == 200
        one_body = one.json()
        assert one_body["run_id"] == run_id
        artifact = client.get(f"/api/v1/runs/{run_id}/artifacts/report_path")
        assert artifact.status_code == 200


def test_api_cancel_endpoint(tmp_path: Path) -> None:
    cfg = _make_ui_config(tmp_path)
    app = create_app(str(cfg))
    with TestClient(app) as client:
        resp = client.post(
            "/api/v1/runs",
            json={
                "mode": "ask",
                "question": "1+1",
                "options": {},
            },
        )
        run_id = resp.json()["run_id"]

        cancel = client.post(f"/api/v1/runs/{run_id}/cancel")
        assert cancel.status_code == 200
        body = cancel.json()
        assert body["run_id"] == run_id
        assert body["cancel_requested"] is True
