from __future__ import annotations

from pathlib import Path

from src.ui.run_store import UIRunStore


def test_run_store_lifecycle(tmp_path: Path) -> None:
    db = tmp_path / "ui_runs.db"
    store = UIRunStore(str(db))
    store.init()

    row = store.create_run(
        run_id="r1",
        mode="research",
        question="q",
        output_path=None,
        details={"request": {"mode": "research", "question": "q"}},
    )
    assert row["status"] == "queued"

    store.mark_running("r1")
    mid = store.get_run("r1")
    assert mid["status"] == "running"
    assert mid["started_at"] is not None

    store.mark_done("r1", output_path="runs/x.md", details={"ok": True})
    done = store.get_run("r1")
    assert done["status"] == "done"
    assert done["output_path"] == "runs/x.md"
    assert done["details"]["ok"] is True

    events = store.list_events_since("r1")
    assert len(events) >= 3


def test_recover_stale_running(tmp_path: Path) -> None:
    db = tmp_path / "ui_runs.db"
    store = UIRunStore(str(db))
    store.init()
    store.create_run(run_id="r2", mode="query", question="x", output_path=None, details={})
    store.mark_running("r2")

    changed = store.recover_stale_running()
    assert changed == 1

    row = store.get_run("r2")
    assert row["status"] == "failed"
    assert row["error_message"] == "stale_after_restart"
    assert Path(db).exists()
