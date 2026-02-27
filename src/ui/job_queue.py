from __future__ import annotations

import asyncio
from typing import Any

from src.ui.executors import UIExecutors
from src.ui.run_store import UIRunStore
from src.ui.settings import UISettings


class UIJobQueue:
    def __init__(self, *, store: UIRunStore, executors: UIExecutors, settings: UISettings):
        self.store = store
        self.executors = executors
        self.settings = settings
        self.queue: asyncio.Queue[str | None] = asyncio.Queue()
        self._workers: list[asyncio.Task[Any]] = []
        self._started = False

    async def start(self) -> None:
        if self._started:
            return
        self._started = True
        worker_count = max(1, int(self.settings.max_workers))
        for idx in range(worker_count):
            self._workers.append(asyncio.create_task(self._worker_loop(idx), name=f"ui-worker-{idx}"))

    async def stop(self) -> None:
        if not self._started:
            return
        for _ in self._workers:
            await self.queue.put(None)
        await asyncio.gather(*self._workers, return_exceptions=True)
        self._workers.clear()
        self._started = False

    async def enqueue(self, run_id: str) -> None:
        await self.queue.put(run_id)

    def status(self) -> dict[str, Any]:
        return {
            "started": self._started,
            "max_workers": len(self._workers),
            "queue_size": int(self.queue.qsize()),
        }

    async def _worker_loop(self, worker_idx: int) -> None:
        while True:
            run_id = await self.queue.get()
            try:
                if run_id is None:
                    return
                await self._execute_one(run_id, worker_idx=worker_idx)
            finally:
                self.queue.task_done()

    async def _execute_one(self, run_id: str, *, worker_idx: int) -> None:
        run = self.store.get_run(run_id)
        if run["status"] != "queued":
            return
        mode = str(run.get("mode", ""))

        if self.store.is_cancel_requested(run_id):
            details = dict(run.get("details") or {})
            self.store.mark_cancelled(run_id, details=details)
            return

        self.store.mark_running(run_id)
        timeout_sec = int(self.settings.job_timeout_sec.get(mode, 3600))

        def emit_event(level: str, message: str, payload: dict[str, Any]) -> None:
            self.store.add_event(
                run_id=run_id,
                level=level,
                message=message,
                payload={**payload, "worker": worker_idx},
            )

        def cancel_requested() -> bool:
            return self.store.is_cancel_requested(run_id)

        try:
            result = await asyncio.wait_for(
                asyncio.to_thread(
                    self.executors.execute,
                    run_id=run_id,
                    request=dict(run.get("details", {}).get("request") or {}),
                    emit_event=emit_event,
                    cancel_requested=cancel_requested,
                ),
                timeout=timeout_sec,
            )
        except asyncio.TimeoutError:
            self.store.mark_failed(run_id, f"run_timeout_after_{timeout_sec}s")
            return
        except Exception as exc:
            message = str(exc)
            if message == "cancelled_before_start":
                details = dict(run.get("details") or {})
                self.store.mark_cancelled(run_id, details=details)
                return
            self.store.mark_failed(run_id, message or exc.__class__.__name__)
            return

        details = dict(run.get("details") or {})
        details.update(result.details)
        if self.store.is_cancel_requested(run_id):
            self.store.mark_cancelled(run_id, details=details)
            return

        self.store.mark_done(run_id, output_path=result.output_path, details=details)
