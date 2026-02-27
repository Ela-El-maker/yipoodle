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
                "chat:",
                "  enabled: true",
                "  default_mode: auto",
                "  max_message_chars: 12000",
                "  max_blob_bytes: 2000000",
                "  sse_replay_limit: 500",
                "  title_autogen: true",
                "  retain_events_days: 30",
                "  db_only_chat_outputs: true",
            ]
        ),
        encoding="utf-8",
    )
    return cfg


def _wait_message_done(client: TestClient, session_id: str, message_id: str, timeout: float = 5.0) -> dict:
    t0 = time.time()
    while True:
        resp = client.get(f"/api/v1/chat/sessions/{session_id}/messages")
        assert resp.status_code == 200
        body = resp.json()
        msg = next((m for m in body["messages"] if m["id"] == message_id), None)
        if msg and msg["status"] in {"done", "failed", "cancelled"}:
            return msg
        if time.time() - t0 > timeout:
            return msg or {}
        time.sleep(0.05)


def test_chat_session_message_flow(tmp_path: Path) -> None:
    cfg = _make_ui_config(tmp_path)
    app = create_app(str(cfg))
    with TestClient(app) as client:
        created = client.post("/api/v1/chat/sessions", json={})
        assert created.status_code == 200
        session_id = created.json()["id"]

        msg_resp = client.post(
            f"/api/v1/chat/sessions/{session_id}/messages",
            json={"content": "23 + 34 = ?", "mode": "ask", "options": {}, "stream": True},
        )
        assert msg_resp.status_code == 200
        body = msg_resp.json()
        assistant_id = body["assistant_message_id"]

        final = _wait_message_done(client, session_id, assistant_id)
        assert final["status"] == "done"
        assert "57" in final["content_markdown"]

        sessions = client.get("/api/v1/chat/sessions")
        assert sessions.status_code == 200
        assert sessions.json()["count"] >= 1


def test_chat_sse_stream_and_final(tmp_path: Path) -> None:
    cfg = _make_ui_config(tmp_path)
    app = create_app(str(cfg))
    with TestClient(app) as client:
        session_id = client.post("/api/v1/chat/sessions", json={}).json()["id"]
        created = client.post(
            f"/api/v1/chat/sessions/{session_id}/messages",
            json={"content": "what is an algorithm", "mode": "ask", "options": {}, "stream": True},
        ).json()
        assistant_id = created["assistant_message_id"]

        seen_event = False
        seen_final = False
        with client.stream("GET", created["events_url"]) as resp:
            assert resp.status_code == 200
            for line in resp.iter_lines():
                if not line:
                    continue
                if line.startswith("event: chat_event"):
                    seen_event = True
                if line.startswith("event: chat_final"):
                    seen_final = True
                    break
        assert seen_event
        assert seen_final

        final = _wait_message_done(client, session_id, assistant_id)
        assert final["status"] in {"done", "failed", "cancelled"}


def test_chat_cancel_message_endpoint(tmp_path: Path) -> None:
    cfg = _make_ui_config(tmp_path)
    app = create_app(str(cfg))
    with TestClient(app) as client:
        session_id = client.post("/api/v1/chat/sessions", json={}).json()["id"]
        created = client.post(
            f"/api/v1/chat/sessions/{session_id}/messages",
            json={"content": "2+2", "mode": "ask", "options": {}, "stream": True},
        ).json()
        assistant_id = created["assistant_message_id"]
        cancel = client.post(f"/api/v1/chat/sessions/{session_id}/cancel/{assistant_id}")
        assert cancel.status_code == 200
        body = cancel.json()
        assert body["message_id"] == assistant_id
        assert body["cancel_requested"] is True


def test_chat_auto_factoid_routes_to_ask(tmp_path: Path) -> None:
    cfg = _make_ui_config(tmp_path)
    app = create_app(str(cfg))
    with TestClient(app) as client:
        session_id = client.post("/api/v1/chat/sessions", json={}).json()["id"]
        created = client.post(
            f"/api/v1/chat/sessions/{session_id}/messages",
            json={"content": "Who is Barack Obama?", "mode": "auto", "options": {}, "stream": True},
        ).json()
        assistant_id = created["assistant_message_id"]
        final = _wait_message_done(client, session_id, assistant_id)
        assert final["status"] in {"done", "failed", "cancelled"}
        if final["status"] == "done":
            assert final["mode_used"] == "ask"
