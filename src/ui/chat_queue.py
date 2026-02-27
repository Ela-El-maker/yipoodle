from __future__ import annotations

import asyncio
from typing import Any

from src.ui.chat_executors import UIChatExecutors
from src.ui.chat_store import UIChatStore
from src.ui.settings import UISettings


class UIChatQueue:
    def __init__(self, *, store: UIChatStore, executors: UIChatExecutors, settings: UISettings):
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
            self._workers.append(asyncio.create_task(self._worker_loop(idx), name=f"ui-chat-worker-{idx}"))

    async def stop(self) -> None:
        if not self._started:
            return
        for _ in self._workers:
            await self.queue.put(None)
        await asyncio.gather(*self._workers, return_exceptions=True)
        self._workers.clear()
        self._started = False

    async def enqueue(self, message_id: str) -> None:
        await self.queue.put(message_id)

    def status(self) -> dict[str, Any]:
        return {
            "started": self._started,
            "max_workers": len(self._workers),
            "queue_size": int(self.queue.qsize()),
        }

    async def _worker_loop(self, worker_idx: int) -> None:
        while True:
            message_id = await self.queue.get()
            try:
                if message_id is None:
                    return
                await self._execute_one(message_id, worker_idx=worker_idx)
            finally:
                self.queue.task_done()

    async def _execute_one(self, message_id: str, *, worker_idx: int) -> None:
        message = self.store.get_message(message_id)
        if message["status"] != "queued":
            return

        request = dict(message.get("metadata", {}).get("request") or {})
        mode_req = str(message.get("mode_requested") or request.get("mode") or "auto")
        timeout_sec = int(self.settings.job_timeout_sec.get(mode_req, 3600))
        session_id = str(message["session_id"])

        if self.store.is_cancel_requested(message_id):
            self.store.mark_cancelled(message_id)
            self.store.add_event(
                session_id=session_id,
                message_id=message_id,
                level="warn",
                event_type="status",
                payload={"status": "cancelled", "reason": "cancelled_before_start"},
            )
            return

        self.store.mark_running(message_id)
        self.store.add_event(
            session_id=session_id,
            message_id=message_id,
            level="info",
            event_type="status",
            payload={"status": "running", "worker": worker_idx},
        )

        def emit_event(level: str, event_type: str, payload: dict[str, Any]) -> None:
            self.store.add_event(
                session_id=session_id,
                message_id=message_id,
                level=level,
                event_type=event_type,
                payload={**(payload or {}), "worker": worker_idx},
            )

        def cancel_requested() -> bool:
            return self.store.is_cancel_requested(message_id)

        try:
            result = await asyncio.wait_for(
                asyncio.to_thread(
                    self.executors.execute,
                    message_id=message_id,
                    request=request,
                    emit_event=emit_event,
                    cancel_requested=cancel_requested,
                ),
                timeout=timeout_sec,
            )
        except asyncio.TimeoutError:
            self.store.mark_failed(message_id, f"run_timeout_after_{timeout_sec}s")
            self.store.add_event(
                session_id=session_id,
                message_id=message_id,
                level="error",
                event_type="error",
                payload={"error": f"run_timeout_after_{timeout_sec}s"},
            )
            return
        except Exception as exc:
            message_error = str(exc) or exc.__class__.__name__
            if message_error == "cancelled_before_start":
                self.store.mark_cancelled(message_id)
                self.store.add_event(
                    session_id=session_id,
                    message_id=message_id,
                    level="warn",
                    event_type="status",
                    payload={"status": "cancelled"},
                )
                return
            self.store.mark_failed(message_id, message_error)
            self.store.add_event(
                session_id=session_id,
                message_id=message_id,
                level="error",
                event_type="error",
                payload={"error": message_error},
            )
            return

        if self.store.is_cancel_requested(message_id):
            self.store.mark_cancelled(message_id, metadata=result.metadata)
            self.store.add_event(
                session_id=session_id,
                message_id=message_id,
                level="warn",
                event_type="status",
                payload={"status": "cancelled"},
            )
            return

        self.store.mark_done(
            message_id,
            mode_used=result.mode_used,
            content_markdown=result.content_markdown,
            metadata=result.metadata,
        )
        for blob in result.blobs:
            self.store.add_blob(
                message_id=message_id,
                blob_type=blob.blob_type,
                blob_text=blob.blob_text,
            )
        self.store.add_event(
            session_id=session_id,
            message_id=message_id,
            level="info",
            event_type="final",
            payload={"status": "done", "mode_used": result.mode_used},
        )
