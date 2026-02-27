from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4
import json

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import StreamingResponse

from src.ui.schemas import (
    ChatMessageCreateRequest,
    ChatMessageCreateResponse,
    ChatMessageRecord,
    ChatSessionCreateRequest,
    ChatSessionRecord,
)


router = APIRouter(prefix="/chat", tags=["chat"])


def _new_session_id() -> str:
    return f"s_{uuid4().hex}"


def _new_message_id() -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"m_{ts}_{uuid4().hex[:10]}"


def _to_session_record(row: dict[str, Any]) -> ChatSessionRecord:
    return ChatSessionRecord.model_validate(row)


def _to_message_record(row: dict[str, Any]) -> ChatMessageRecord:
    return ChatMessageRecord.model_validate(row)


def _title_from_content(text: str) -> str:
    cleaned = " ".join((text or "").strip().split())
    if not cleaned:
        return "New chat"
    return cleaned[:80]


@router.post("/sessions", response_model=ChatSessionRecord)
def api_create_session(request: Request, body: ChatSessionCreateRequest) -> ChatSessionRecord:
    service = request.app.state.ui_service
    session_id = _new_session_id()
    session = service.chat_store.create_session(session_id=session_id, title=body.title)
    return _to_session_record(session)


@router.get("/sessions")
def api_list_sessions(
    request: Request,
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
) -> dict[str, Any]:
    rows = request.app.state.ui_service.chat_store.list_sessions(limit=limit, offset=offset)
    return {
        "count": len(rows),
        "sessions": [r.model_dump(mode="json") for r in [_to_session_record(x) for x in rows]],
    }


@router.get("/sessions/{session_id}", response_model=ChatSessionRecord)
def api_get_session(request: Request, session_id: str) -> ChatSessionRecord:
    try:
        row = request.app.state.ui_service.chat_store.get_session(session_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return _to_session_record(row)


@router.post("/sessions/{session_id}/messages", response_model=ChatMessageCreateResponse)
async def api_create_message(
    request: Request,
    session_id: str,
    body: ChatMessageCreateRequest,
) -> ChatMessageCreateResponse:
    service = request.app.state.ui_service
    settings = service.settings
    content = (body.content or "").strip()
    if not content:
        raise HTTPException(status_code=400, detail="content is required")
    if len(content) > int(settings.chat_max_message_chars):
        raise HTTPException(status_code=400, detail=f"content exceeds max_message_chars={settings.chat_max_message_chars}")

    try:
        session = service.chat_store.get_session(session_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    if settings.chat_title_autogen and not (session.get("title") or "").strip():
        service.chat_store.update_session_title(session_id=session_id, title=_title_from_content(content))

    user_message_id = _new_message_id()
    assistant_message_id = _new_message_id()
    mode_requested = str(body.mode)
    request_payload = {
        "content": content,
        "mode": mode_requested,
        "options": body.options or {},
        "stream": bool(body.stream),
    }

    service.chat_store.create_message(
        message_id=user_message_id,
        session_id=session_id,
        role="user",
        mode_requested=mode_requested,
        status="done",
        content_markdown=content,
        metadata={"source": "chat_ui"},
    )
    service.chat_store.create_message(
        message_id=assistant_message_id,
        session_id=session_id,
        role="assistant",
        mode_requested=mode_requested,
        status="queued",
        content_markdown="",
        metadata={
            "request": request_payload,
            "user_message_id": user_message_id,
        },
    )
    service.chat_store.add_event(
        session_id=session_id,
        message_id=assistant_message_id,
        level="info",
        event_type="status",
        payload={"status": "queued"},
    )
    await service.chat_queue.enqueue(assistant_message_id)

    return ChatMessageCreateResponse(
        session_id=session_id,
        message_id=user_message_id,
        assistant_message_id=assistant_message_id,
        status="queued",
        events_url=f"/api/v1/chat/sessions/{session_id}/events?message_id={assistant_message_id}",
    )


@router.get("/sessions/{session_id}/messages")
def api_list_messages(
    request: Request,
    session_id: str,
    limit: int = Query(default=500, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
    with_blobs: bool = Query(default=False),
) -> dict[str, Any]:
    store = request.app.state.ui_service.chat_store
    try:
        store.get_session(session_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    messages = store.list_messages(session_id, limit=limit, offset=offset)
    rows = [_to_message_record(m).model_dump(mode="json") for m in messages]
    if with_blobs:
        for msg in rows:
            if msg["role"] != "assistant":
                continue
            blobs = store.list_blobs(msg["id"])
            msg["blobs"] = [
                {
                    "id": b["id"],
                    "blob_type": b["blob_type"],
                    "created_at": b["created_at"],
                    "size": len((b.get("blob_text") or "").encode("utf-8", errors="replace")),
                }
                for b in blobs
            ]
    return {"count": len(rows), "messages": rows}


@router.post("/sessions/{session_id}/cancel/{message_id}")
def api_cancel_message(request: Request, session_id: str, message_id: str) -> dict[str, Any]:
    store = request.app.state.ui_service.chat_store
    try:
        msg = store.get_message(message_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    if str(msg["session_id"]) != session_id:
        raise HTTPException(status_code=404, detail="message not found in session")
    accepted = store.request_cancel(message_id)
    store.add_event(
        session_id=session_id,
        message_id=message_id,
        level="warn",
        event_type="status",
        payload={"status": "cancel_requested", "accepted": accepted},
    )
    latest = store.get_message(message_id)
    return {
        "session_id": session_id,
        "message_id": message_id,
        "accepted": accepted,
        "status": latest["status"],
        "cancel_requested": latest["cancel_requested"],
    }


@router.get("/sessions/{session_id}/events")
async def api_session_events(
    request: Request,
    session_id: str,
    message_id: str | None = Query(default=None),
    after_seq: int = Query(default=0, ge=0),
) -> StreamingResponse:
    service = request.app.state.ui_service
    store = service.chat_store
    keepalive_sec = max(1, int(service.settings.sse_keepalive_sec))
    replay_limit = max(1, int(service.settings.chat_sse_replay_limit))
    try:
        store.get_session(session_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    async def _stream():
        seq = int(after_seq)
        idle_ticks = 0
        while True:
            events = store.list_events_since(
                session_id,
                after_seq=seq,
                limit=min(200, replay_limit),
                message_id=message_id,
            )
            if events:
                for ev in events:
                    seq = max(seq, int(ev["seq"]))
                    payload = json.dumps(ev, ensure_ascii=True)
                    yield f"event: chat_event\ndata: {payload}\n\n"
                idle_ticks = 0
            else:
                idle_ticks += 1
                if idle_ticks >= keepalive_sec:
                    idle_ticks = 0
                    yield "event: keepalive\ndata: {}\n\n"

            if message_id:
                try:
                    msg = store.get_message(message_id)
                except KeyError:
                    yield "event: error\ndata: {\"error\":\"message_not_found\"}\n\n"
                    return
                if msg["status"] in {"done", "failed", "cancelled"} and not events:
                    final_payload = json.dumps(
                        {
                            "session_id": session_id,
                            "message_id": message_id,
                            "status": msg["status"],
                        },
                        ensure_ascii=True,
                    )
                    yield f"event: chat_final\ndata: {final_payload}\n\n"
                    return
            await asyncio.sleep(1.0)

    return StreamingResponse(_stream(), media_type="text/event-stream")
