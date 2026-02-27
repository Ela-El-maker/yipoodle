from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

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
                "  - runs/query",
                "sse_keepalive_sec: 1",
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


def test_sse_emits_events_and_final(tmp_path: Path) -> None:
    cfg = _make_ui_config(tmp_path)
    app = create_app(str(cfg))

    with TestClient(app) as client:
        created = client.post("/api/v1/runs", json={"mode": "ask", "question": "2+2", "options": {}}).json()
        run_id = created["run_id"]

        saw_event = False
        saw_final = False
        with client.stream("GET", f"/api/v1/runs/{run_id}/events") as resp:
            assert resp.status_code == 200
            for line in resp.iter_lines():
                if not line:
                    continue
                txt = line.decode() if isinstance(line, bytes) else str(line)
                if "event: run_event" in txt:
                    saw_event = True
                if "event: run_final" in txt:
                    saw_final = True
                    break

        assert saw_event is True
        assert saw_final is True
