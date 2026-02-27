from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4
import json

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import FileResponse, StreamingResponse

from src.ui.artifacts import resolve_artifact_path
from src.ui.schemas import RunCreateRequest, RunCreateResponse, RunRecord


router = APIRouter(prefix="/runs", tags=["runs"])


def _new_run_id() -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{ts}_{uuid4().hex[:8]}"


def _to_run_record(row: dict[str, Any]) -> RunRecord:
    return RunRecord.model_validate(row)


@router.post("", response_model=RunCreateResponse)
async def api_create_run(request: Request, body: RunCreateRequest) -> RunCreateResponse:
    service = request.app.state.ui_service

    if body.mode != "automation" and not (body.question or "").strip():
        raise HTTPException(status_code=400, detail="question is required for this mode")

    run_id = _new_run_id()
    req_payload = body.model_dump()
    details = {
        "request": req_payload,
    }
    created = service.store.create_run(
        run_id=run_id,
        mode=body.mode,
        question=body.question,
        output_path=body.output_path,
        details=details,
    )
    await service.job_queue.enqueue(run_id)

    return RunCreateResponse(
        run_id=run_id,
        status=created["status"],
        created_at=created["created_at"],
        mode=body.mode,
        question=body.question,
        status_url=f"/api/v1/runs/{run_id}",
        events_url=f"/api/v1/runs/{run_id}/events",
    )


@router.get("")
def api_list_runs(
    request: Request,
    status: str | None = Query(default=None),
    mode: str | None = Query(default=None),
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
) -> dict[str, Any]:
    rows = request.app.state.ui_service.store.list_runs(
        status=status,
        mode=mode,
        limit=limit,
        offset=offset,
    )
    return {
        "count": len(rows),
        "runs": [r.model_dump(mode="json") for r in [_to_run_record(x) for x in rows]],
    }


@router.get("/{run_id}", response_model=RunRecord)
def api_get_run(request: Request, run_id: str) -> RunRecord:
    try:
        row = request.app.state.ui_service.store.get_run(run_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return _to_run_record(row)


@router.post("/{run_id}/cancel")
def api_cancel_run(request: Request, run_id: str) -> dict[str, Any]:
    store = request.app.state.ui_service.store
    try:
        accepted = store.request_cancel(run_id)
        run = store.get_run(run_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"run_id": run_id, "accepted": accepted, "status": run["status"], "cancel_requested": run["cancel_requested"]}


@router.get("/{run_id}/events")
async def api_run_events(
    request: Request,
    run_id: str,
    after_seq: int = Query(default=0, ge=0),
) -> StreamingResponse:
    store = request.app.state.ui_service.store
    keepalive_sec = max(1, int(request.app.state.ui_service.settings.sse_keepalive_sec))

    try:
        store.get_run(run_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    async def _stream():
        seq = int(after_seq)
        idle_ticks = 0
        while True:
            try:
                run = store.get_run(run_id)
            except KeyError:
                yield "event: error\ndata: {\"error\":\"run_not_found\"}\n\n"
                return

            events = store.list_events_since(run_id, after_seq=seq, limit=200)
            if events:
                for ev in events:
                    seq = max(seq, int(ev["seq"]))
                    payload = json.dumps(ev, ensure_ascii=True)
                    yield f"event: run_event\ndata: {payload}\n\n"
                idle_ticks = 0
            else:
                idle_ticks += 1
                if idle_ticks >= keepalive_sec:
                    idle_ticks = 0
                    yield "event: keepalive\ndata: {}\n\n"

            if run["status"] in {"done", "failed", "cancelled"} and not events:
                final_payload = json.dumps({"run_id": run_id, "status": run["status"]}, ensure_ascii=True)
                yield f"event: run_final\ndata: {final_payload}\n\n"
                return
            await asyncio.sleep(1.0)

    return StreamingResponse(_stream(), media_type="text/event-stream")


@router.get("/{run_id}/artifacts/{artifact_key}")
def api_get_artifact(request: Request, run_id: str, artifact_key: str) -> FileResponse:
    service = request.app.state.ui_service
    try:
        run = service.store.get_run(run_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    try:
        p = resolve_artifact_path(run_details=run.get("details", {}), artifact_key=artifact_key, roots=service.settings.artifacts_roots)
    except (KeyError, FileNotFoundError, PermissionError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    media = "application/octet-stream"
    suffix = p.suffix.lower()
    if suffix == ".json":
        media = "application/json"
    elif suffix == ".md":
        media = "text/markdown; charset=utf-8"
    elif suffix in {".txt", ".log"}:
        media = "text/plain; charset=utf-8"
    return FileResponse(path=str(p), media_type=media, filename=p.name)
