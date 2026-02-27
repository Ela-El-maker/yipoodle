from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from src.ui.app import create_app
from src.ui.chat_store import UIChatStore


def _make_ui_config(tmp_path: Path) -> Path:
    cfg = tmp_path / "ui.yaml"
    cfg.write_text(
        "\n".join(
            [
                "max_workers: 1",
                f"run_db_path: {tmp_path / 'ui_runs.db'}",
                "artifacts_roots:",
                f"  - {tmp_path}",
                "chat:",
                "  enabled: true",
                "  retain_events_days: 30",
            ]
        ),
        encoding="utf-8",
    )
    return cfg


def test_chat_running_message_marked_stale_after_restart(tmp_path: Path) -> None:
    db = tmp_path / "ui_runs.db"
    store = UIChatStore(str(db))
    store.init()
    store.create_session(session_id="s1", title="x")
    store.create_message(
        message_id="m1",
        session_id="s1",
        role="assistant",
        mode_requested="ask",
        status="queued",
        content_markdown="",
        metadata={"request": {"content": "2+2", "mode": "ask", "options": {}}},
    )
    store.mark_running("m1")

    app = create_app(str(_make_ui_config(tmp_path)))
    with TestClient(app):
        pass

    store2 = UIChatStore(str(db))
    msg = store2.get_message("m1")
    assert msg["status"] == "failed"
    assert msg["error_message"] == "stale_after_restart"
