from __future__ import annotations

from pathlib import Path
from typing import Any
import json
import urllib.request

from fastapi import APIRouter, Request

from src.apps.automation import load_automation_config


router = APIRouter(tags=["health"])


def _vector_health(endpoint: str | None) -> dict[str, Any]:
    if not endpoint:
        return {"configured": False, "ok": None}
    url = endpoint.rstrip("/") + "/health"
    try:
        with urllib.request.urlopen(url, timeout=3.0) as resp:
            body = resp.read().decode("utf-8", errors="replace")
        payload = json.loads(body)
        if isinstance(payload, dict):
            return {"configured": True, "ok": bool(payload.get("ok", True)), "payload": payload}
        return {"configured": True, "ok": False, "payload": {"raw": body[:300]}}
    except Exception as exc:
        return {"configured": True, "ok": False, "error": str(exc)}


@router.get("/health")
def api_health(request: Request) -> dict[str, Any]:
    service = request.app.state.ui_service
    settings = service.settings

    digest_path = None
    digest_count = 0
    automation_summary = None
    try:
        cfg = load_automation_config(settings.automation_config)
        digest_path = str(((cfg.get("monitoring", {}) or {}).get("digest", {}) or {}).get("path", "runs/audit/monitor_digest_queue.json"))
        dp = Path(digest_path)
        if dp.exists():
            payload = json.loads(dp.read_text(encoding="utf-8"))
            if isinstance(payload, list):
                digest_count = len(payload)

        audit_dir = str((cfg.get("paths", {}) or {}).get("audit_dir", "runs/audit"))
        summary_path = Path(audit_dir) / "latest_summary.json"
        if summary_path.exists():
            automation_summary = str(summary_path)
    except Exception:
        pass

    return {
        "ok": True,
        "queue": service.job_queue.status(),
        "chat_queue": service.chat_queue.status(),
        "run_db_path": settings.run_db_path,
        "digest_queue_path": digest_path,
        "digest_queue_count": digest_count,
        "latest_automation_summary": automation_summary,
        "vector_service": _vector_health(settings.vector_service_endpoint),
    }
