from __future__ import annotations

import asyncio
from pathlib import Path

from src.ui.executors import ExecuteResult
from src.ui.job_queue import UIJobQueue
from src.ui.run_store import UIRunStore
from src.ui.settings import UISettings


class _FakeExecutors:
    def execute(self, *, run_id, request, emit_event, cancel_requested):
        emit_event("info", "fake_exec", {"run_id": run_id})
        return ExecuteResult(output_path="runs/query/out.md", details={"mode": request.get("mode")})


def _wait_status(store: UIRunStore, run_id: str, target: set[str], timeout: float = 3.0) -> str:
    async def _inner() -> str:
        t0 = asyncio.get_event_loop().time()
        while True:
            row = store.get_run(run_id)
            if row["status"] in target:
                return row["status"]
            if (asyncio.get_event_loop().time() - t0) > timeout:
                return row["status"]
            await asyncio.sleep(0.05)

    return asyncio.run(_inner())


def test_job_queue_executes_and_marks_done(tmp_path: Path) -> None:
    store = UIRunStore(str(tmp_path / "runs.db"))
    store.init()
    store.create_run(
        run_id="r1",
        mode="ask",
        question="q",
        output_path=None,
        details={"request": {"mode": "ask", "question": "q"}},
    )

    settings = UISettings.model_validate({
        "run_db_path": str(tmp_path / "runs.db"),
        "max_workers": 1,
        "artifacts_roots": [str(tmp_path)],
    })
    queue = UIJobQueue(store=store, executors=_FakeExecutors(), settings=settings)

    async def _run():
        await queue.start()
        await queue.enqueue("r1")
        await asyncio.sleep(0.2)
        await queue.stop()

    asyncio.run(_run())
    status = _wait_status(store, "r1", {"done", "failed", "cancelled"})
    assert status == "done"


def test_job_queue_respects_cancel_before_start(tmp_path: Path) -> None:
    store = UIRunStore(str(tmp_path / "runs.db"))
    store.init()
    store.create_run(
        run_id="r2",
        mode="ask",
        question="q",
        output_path=None,
        details={"request": {"mode": "ask", "question": "q"}},
    )
    store.request_cancel("r2")

    settings = UISettings.model_validate({
        "run_db_path": str(tmp_path / "runs.db"),
        "max_workers": 1,
        "artifacts_roots": [str(tmp_path)],
    })
    queue = UIJobQueue(store=store, executors=_FakeExecutors(), settings=settings)

    async def _run():
        await queue.start()
        await queue.enqueue("r2")
        await asyncio.sleep(0.2)
        await queue.stop()

    asyncio.run(_run())
    row = store.get_run("r2")
    assert row["status"] == "cancelled"
