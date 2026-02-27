from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4
import json

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from src.apps.monitor_mode import unregister_monitor


router = APIRouter(prefix="/monitor", tags=["monitor"])


class MonitorCreateRequest(BaseModel):
    question: str
    schedule: str | None = None
    register_schedule: bool = True
    schedule_backend: str = "auto"
    automation_config: str | None = None


@router.get("/topics")
def api_monitor_topics(_request: Request) -> dict[str, Any]:
    topics_dir = Path("runs/monitor/topics")
    rows: list[dict[str, Any]] = []
    if topics_dir.exists():
        for p in sorted(topics_dir.glob("*.json")):
            try:
                payload = json.loads(p.read_text(encoding="utf-8"))
                if isinstance(payload, dict):
                    payload["_path"] = str(p)
                    rows.append(payload)
            except Exception:
                rows.append({"name": p.stem, "_path": str(p), "_parse_error": True})
    return {"count": len(rows), "topics": rows}


@router.post("/topics")
async def api_monitor_create(request: Request, body: MonitorCreateRequest) -> dict[str, Any]:
    service = request.app.state.ui_service
    run_id = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{uuid4().hex[:8]}"
    req_payload = {
        "mode": "monitor",
        "question": body.question,
        "index": None,
        "sources_config": service.settings.sources_config,
        "automation_config": body.automation_config or service.settings.automation_config,
        "output_path": None,
        "options": {
            "schedule": body.schedule,
            "register_schedule": body.register_schedule,
            "schedule_backend": body.schedule_backend,
        },
    }
    created = service.store.create_run(
        run_id=run_id,
        mode="monitor",
        question=body.question,
        output_path=None,
        details={"request": req_payload},
    )
    await service.job_queue.enqueue(run_id)
    return {
        "run_id": run_id,
        "status": created["status"],
        "status_url": f"/api/v1/runs/{run_id}",
        "events_url": f"/api/v1/runs/{run_id}/events",
    }


@router.post("/topics/{name}/unregister")
def api_monitor_unregister(name: str, delete_files: bool = False) -> dict[str, Any]:
    try:
        payload = unregister_monitor(name_or_question=name, delete_files=bool(delete_files))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return payload
