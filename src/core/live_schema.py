from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


class LiveSourceConfig(BaseModel):
    name: str
    enabled: bool = False
    type: Literal["rest", "rss", "scrape"]
    endpoint: str | None = None
    url: str | None = None
    method: Literal["GET", "POST"] = "GET"
    query_params: dict[str, Any] = Field(default_factory=dict)
    auth: dict[str, Any] = Field(default_factory=dict)
    rate_limit_rpm: int = 30
    cache_ttl_sec: int = 300
    timeout_sec: int = 20
    transform: dict[str, Any] = Field(default_factory=dict)
    domain_tags: list[str] = Field(default_factory=list)
    source_trust: float = 1.0
    selector: str | None = None
    limit: int | None = None


class LiveItem(BaseModel):
    id: str
    url: str
    title: str | None = None
    text: str
    published_at: str | None = None
    source: str
    raw_meta: dict[str, Any] = Field(default_factory=dict)


class SnapshotItem(BaseModel):
    snapshot_id: str
    source: str
    query: str
    retrieved_at: datetime
    content_hash: str
    cache_key: str
    items: list[LiveItem]
    raw_path: str | None = None

