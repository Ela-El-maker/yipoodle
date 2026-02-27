from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Query, Request

from src.apps.kb_query import run_kb_query


router = APIRouter(prefix="/kb", tags=["kb"])


@router.get("/query")
def api_kb_query(
    request: Request,
    query: str = Query(..., min_length=1),
    topic: str | None = Query(default=None),
    top_k: int = Query(default=10, ge=1, le=100),
) -> dict[str, Any]:
    settings = request.app.state.ui_service.settings
    payload = run_kb_query(
        kb_db=settings.kb_db,
        query=query,
        topic=topic,
        top_k=int(top_k),
    )
    return payload
