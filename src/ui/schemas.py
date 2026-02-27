from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


RunMode = Literal["ask", "research", "monitor", "notes", "query", "automation"]
RunStatus = Literal["queued", "running", "done", "failed", "cancelled"]
ChatMode = Literal["auto", "ask", "query", "research", "notes", "monitor", "automation"]
ChatMessageRole = Literal["user", "assistant", "system"]


class RunCreateRequest(BaseModel):
    mode: RunMode
    question: str | None = None
    index: str | None = None
    sources_config: str = "config/sources.yaml"
    automation_config: str = "config/automation.yaml"
    output_path: str | None = None
    options: dict[str, Any] = Field(default_factory=dict)


class RunCreateResponse(BaseModel):
    run_id: str
    status: RunStatus
    created_at: datetime
    mode: RunMode
    question: str | None = None
    status_url: str
    events_url: str


class RunRecord(BaseModel):
    run_id: str
    mode: str
    question: str | None = None
    status: str
    created_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None
    error_message: str | None = None
    output_path: str | None = None
    details: dict[str, Any] = Field(default_factory=dict)
    cancel_requested: bool = False


class RunEvent(BaseModel):
    seq: int
    run_id: str
    created_at: datetime
    level: str
    message: str
    payload: dict[str, Any] = Field(default_factory=dict)


class ChatSessionCreateRequest(BaseModel):
    title: str | None = None


class ChatSessionRecord(BaseModel):
    id: str
    title: str | None = None
    created_at: datetime
    updated_at: datetime
    archived: bool = False


class ChatMessageCreateRequest(BaseModel):
    content: str
    mode: ChatMode = "auto"
    options: dict[str, Any] = Field(default_factory=dict)
    stream: bool = True


class ChatMessageRecord(BaseModel):
    id: str
    session_id: str
    role: ChatMessageRole
    mode_requested: str | None = None
    mode_used: str | None = None
    status: RunStatus
    content_markdown: str = ""
    error_message: str | None = None
    created_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None
    run_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    cancel_requested: bool = False


class ChatMessageCreateResponse(BaseModel):
    session_id: str
    message_id: str
    assistant_message_id: str
    status: RunStatus
    events_url: str


class ChatEventRecord(BaseModel):
    seq: int
    session_id: str
    message_id: str
    created_at: datetime
    level: str
    event_type: str
    payload: dict[str, Any] = Field(default_factory=dict)
