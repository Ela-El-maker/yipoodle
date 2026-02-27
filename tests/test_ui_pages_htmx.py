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
                "  - runs/research_reports",
                "  - runs/monitor",
                "  - runs/notes",
                "  - runs/audit/runs",
            ]
        ),
        encoding="utf-8",
    )
    return cfg


def test_pages_render(tmp_path: Path) -> None:
    cfg = _make_ui_config(tmp_path)
    app = create_app(str(cfg))
    with TestClient(app) as client:
        r1 = client.get("/")
        assert r1.status_code == 200
        assert "Sessions" in r1.text

        r0 = client.get("/run-console")
        assert r0.status_code == 200
        assert "Run Console" in r0.text

        r2 = client.get("/runs")
        assert r2.status_code == 200
        assert "Run History" in r2.text

        r3 = client.get("/monitor")
        assert r3.status_code == 200
        assert "Monitoring Dashboard" in r3.text

        r4 = client.get("/health")
        assert r4.status_code == 200
        assert "System Health" in r4.text
