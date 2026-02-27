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
                f"kb_db: {tmp_path / 'knowledge.db'}",
            ]
        ),
        encoding="utf-8",
    )
    return cfg


def test_kb_query_endpoint(tmp_path: Path, monkeypatch) -> None:
    from src.ui.routes import kb as kb_routes

    monkeypatch.setattr(
        kb_routes,
        "run_kb_query",
        lambda kb_db, query, topic, top_k: {
            "kb_db": kb_db,
            "query": query,
            "topic": topic,
            "top_k": top_k,
            "count": 1,
            "claims": [{"claim_text": "x"}],
        },
    )

    cfg = _make_ui_config(tmp_path)
    app = create_app(str(cfg))
    with TestClient(app) as client:
        resp = client.get("/api/v1/kb/query", params={"query": "boundary failure modes", "topic": "cv", "top_k": 5})
        assert resp.status_code == 200
        body = resp.json()
        assert body["count"] == 1
        assert body["query"] == "boundary failure modes"
