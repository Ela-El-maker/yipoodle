from __future__ import annotations

from datetime import datetime, timedelta, timezone
from hashlib import sha256
from pathlib import Path
import json
import re
from typing import Any

from src.core.live_schema import LiveItem, SnapshotItem
from src.core.schemas import SnippetRecord


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _safe_source(source: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", source.strip()) or "live"


def _safe_key(source: str, query: str, params: dict[str, Any]) -> str:
    payload = {"source": source, "query": query, "params": params}
    return sha256(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def _snapshot_hash(source: str, query: str, items: list[LiveItem]) -> str:
    body = {"source": source, "query": query, "items": [i.model_dump() for i in items]}
    return sha256(json.dumps(body, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def _strip_html(text: str) -> str:
    t = re.sub(r"(?is)<script.*?>.*?</script>", " ", text or "")
    t = re.sub(r"(?is)<style.*?>.*?</style>", " ", t)
    t = re.sub(r"(?s)<[^>]+>", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _cache_file(root_dir: str, source: str, cache_key: str) -> Path:
    return Path(root_dir) / _safe_source(source) / f"{cache_key}.snapshot.json"


def load_cached_snapshot(
    *,
    root_dir: str,
    source: str,
    query: str,
    params: dict[str, Any],
    ttl_sec: int,
) -> SnapshotItem | None:
    if int(ttl_sec) <= 0:
        return None
    key = _safe_key(source, query, params)
    path = _cache_file(root_dir, source, key)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        snap = SnapshotItem(**payload)
    except Exception:
        return None
    age = _now_utc() - snap.retrieved_at
    if age <= timedelta(seconds=int(ttl_sec)):
        return snap
    return None


def save_snapshot(
    *,
    root_dir: str,
    source: str,
    query: str,
    params: dict[str, Any],
    items: list[LiveItem],
    persist_raw: bool = True,
    raw_payload: str | None = None,
) -> tuple[SnapshotItem, str]:
    key = _safe_key(source, query, params)
    content_hash = _snapshot_hash(source, query, items)
    snapshot_id = content_hash[:16]
    retrieved_at = _now_utc()
    source_dir = Path(root_dir) / _safe_source(source)
    source_dir.mkdir(parents=True, exist_ok=True)

    raw_path: str | None = None
    if persist_raw and raw_payload is not None:
        raw_file = source_dir / f"{key}.raw.txt"
        raw_file.write_text(raw_payload, encoding="utf-8")
        raw_path = str(raw_file)

    snap = SnapshotItem(
        snapshot_id=snapshot_id,
        source=source,
        query=query,
        retrieved_at=retrieved_at,
        content_hash=content_hash,
        cache_key=key,
        items=items,
        raw_path=raw_path,
    )
    path = _cache_file(root_dir, source, key)
    path.write_text(json.dumps(snap.model_dump(mode="json"), indent=2), encoding="utf-8")
    return snap, str(path)


def snapshot_to_snippets(
    snap: SnapshotItem,
    *,
    max_items: int = 20,
    section: str = "live",
) -> list[SnippetRecord]:
    rows: list[SnippetRecord] = []
    paper_id = f"SNAP:{snap.snapshot_id}"
    for i, item in enumerate(snap.items[: max(1, int(max_items))], start=1):
        text = _strip_html(item.text)
        if not text:
            continue
        sid = f"SNAP:{snap.snapshot_id}:S{i}"
        rows.append(
            SnippetRecord(
                snippet_id=sid,
                paper_id=paper_id,
                section=section,
                text=text,
                page_hint=None,
                token_count=max(1, len(text.split())),
                paper_year=None,
                paper_venue=item.source,
                citation_count=0,
                extraction_quality_score=1.0,
                extraction_quality_band="good",
                extraction_source="native",
            )
        )
    return rows
