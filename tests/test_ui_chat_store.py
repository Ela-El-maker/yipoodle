from __future__ import annotations

from pathlib import Path

from src.ui.chat_store import UIChatStore


def test_chat_store_session_message_blob_event_lifecycle(tmp_path: Path) -> None:
    db = tmp_path / "ui_runs.db"
    store = UIChatStore(str(db))
    store.init()

    sess = store.create_session(session_id="s1", title="Chat")
    assert sess["id"] == "s1"

    user = store.create_message(
        message_id="m1",
        session_id="s1",
        role="user",
        mode_requested="auto",
        status="done",
        content_markdown="hello",
        metadata={"x": 1},
    )
    assert user["role"] == "user"

    assistant = store.create_message(
        message_id="m2",
        session_id="s1",
        role="assistant",
        mode_requested="ask",
        status="queued",
        content_markdown="",
        metadata={"request": {"content": "2+2", "mode": "ask", "options": {}}},
    )
    assert assistant["status"] == "queued"

    store.mark_running("m2")
    store.add_event(session_id="s1", message_id="m2", level="info", event_type="progress", payload={"step": 1})
    store.add_blob(message_id="m2", blob_type="report_md", blob_text="# Ask\n\n4\n")
    store.mark_done("m2", mode_used="ask", content_markdown="# Ask\n\n4\n", metadata={"ok": True})

    rows = store.list_messages("s1")
    assert len(rows) == 2
    assert rows[-1]["status"] == "done"
    assert rows[-1]["mode_used"] == "ask"

    blobs = store.list_blobs("m2")
    assert len(blobs) == 1
    assert blobs[0]["blob_type"] == "report_md"

    events = store.list_events_since("s1")
    assert len(events) >= 1


def test_chat_store_recover_stale_running(tmp_path: Path) -> None:
    db = tmp_path / "ui_runs.db"
    store = UIChatStore(str(db))
    store.init()
    store.create_session(session_id="s1", title=None)
    store.create_message(
        message_id="m1",
        session_id="s1",
        role="assistant",
        mode_requested="query",
        status="queued",
        content_markdown="",
        metadata={},
    )
    store.mark_running("m1")
    changed = store.recover_stale_running()
    assert changed == 1
    row = store.get_message("m1")
    assert row["status"] == "failed"
    assert row["error_message"] == "stale_after_restart"
