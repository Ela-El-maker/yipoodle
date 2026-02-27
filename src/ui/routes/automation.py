from __future__ import annotations

from pathlib import Path
from typing import Any
import json

from fastapi import APIRouter, HTTPException, Request

from src.apps.automation import load_automation_config


router = APIRouter(prefix="/automation", tags=["automation"])


@router.get("/latest-summary")
def api_latest_summary(request: Request) -> dict[str, Any]:
    settings = request.app.state.ui_service.settings
    try:
        cfg = load_automation_config(settings.automation_config)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"failed to load automation config: {exc}") from exc

    audit_dir = str((cfg.get("paths", {}) or {}).get("audit_dir", "runs/audit"))
    summary_path = Path(audit_dir) / "latest_summary.json"
    if not summary_path.exists():
        raise HTTPException(status_code=404, detail=f"summary not found: {summary_path}")

    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    return {
        "path": str(summary_path),
        "summary": payload,
    }
