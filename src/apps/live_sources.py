from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import json
import re
import time
from typing import Any
import xml.etree.ElementTree as ET

import requests

from src.core.live_schema import LiveItem, LiveSourceConfig

_LAST_CALL: dict[str, float] = {}


def _safe_int(value: Any, default: int, min_v: int) -> int:
    try:
        return max(min_v, int(value))
    except Exception:
        return max(min_v, int(default))


def _safe_float(value: Any, default: float, min_v: float) -> float:
    try:
        return max(min_v, float(value))
    except Exception:
        return max(min_v, float(default))


def live_sources_config(cfg: dict[str, Any]) -> dict[str, LiveSourceConfig]:
    block = cfg.get("live_sources", {}) if isinstance(cfg, dict) else {}
    if not isinstance(block, dict):
        return {}
    out: dict[str, LiveSourceConfig] = {}
    for name, raw in block.items():
        if not isinstance(raw, dict):
            continue
        typ = str(raw.get("type") or "").strip().lower()
        if typ not in {"rest", "rss", "scrape"}:
            continue
        cfg_row = LiveSourceConfig(
            name=str(name),
            enabled=bool(raw.get("enabled", False)),
            type=typ,  # type: ignore[arg-type]
            endpoint=str(raw.get("endpoint")) if raw.get("endpoint") else None,
            url=str(raw.get("url")) if raw.get("url") else None,
            method=str(raw.get("method", "GET")).upper(),  # type: ignore[arg-type]
            query_params=(raw.get("query_params") if isinstance(raw.get("query_params"), dict) else {}),
            auth=(raw.get("auth") if isinstance(raw.get("auth"), dict) else {}),
            rate_limit_rpm=_safe_int(raw.get("rate_limit_rpm"), 30, 1),
            cache_ttl_sec=_safe_int(raw.get("cache_ttl_sec"), 300, 0),
            timeout_sec=_safe_int(raw.get("timeout_sec"), 20, 1),
            transform=(raw.get("transform") if isinstance(raw.get("transform"), dict) else {}),
            domain_tags=[str(x).strip().lower() for x in (raw.get("domain_tags") or []) if str(x).strip()],
            source_trust=_safe_float(raw.get("source_trust"), 1.0, 0.1),
            selector=str(raw.get("selector")) if raw.get("selector") else None,
            limit=_safe_int(raw.get("limit"), 20, 1) if raw.get("limit") is not None else None,
        )
        out[cfg_row.name] = cfg_row
    return out


def live_snapshot_config(cfg: dict[str, Any]) -> dict[str, Any]:
    block = cfg.get("live_snapshot", {}) if isinstance(cfg, dict) else {}
    if not isinstance(block, dict):
        block = {}
    return {
        "root_dir": str(block.get("root_dir") or "data/live_snapshots"),
        "retention_days": _safe_int(block.get("retention_days"), 30, 1),
        "persist_raw": bool(block.get("persist_raw", True)),
        "max_body_bytes": _safe_int(block.get("max_body_bytes"), 2_000_000, 1024),
    }


def _apply_rate_limit(source_name: str, rpm: int) -> None:
    if rpm <= 0:
        return
    min_gap = 60.0 / float(rpm)
    now = time.monotonic()
    last = _LAST_CALL.get(source_name)
    if last is not None:
        delta = now - last
        if delta < min_gap:
            time.sleep(min_gap - delta)
    _LAST_CALL[source_name] = time.monotonic()


def _headers_from_auth(auth: dict[str, Any]) -> dict[str, str]:
    if not isinstance(auth, dict):
        return {}
    typ = str(auth.get("type") or "none").strip().lower()
    if typ == "bearer":
        token = str(auth.get("token") or "").strip()
        return {"Authorization": f"Bearer {token}"} if token else {}
    if typ == "header":
        name = str(auth.get("name") or "").strip()
        value = str(auth.get("value") or "").strip()
        return {name: value} if name and value else {}
    return {}


def _path_get(payload: Any, path: str) -> Any:
    cur = payload
    for token in re.finditer(r"([A-Za-z0-9_]+)|\[(\d+)\]", path):
        key = token.group(1)
        idx = token.group(2)
        if key is not None:
            if not isinstance(cur, dict):
                return None
            cur = cur.get(key)
        elif idx is not None:
            if not isinstance(cur, list):
                return None
            i = int(idx)
            if i < 0 or i >= len(cur):
                return None
            cur = cur[i]
    return cur


def _id_for(source: str, text: str, url: str) -> str:
    base = f"{source}|{url}|{text[:120]}"
    return sha256(base.encode("utf-8")).hexdigest()[:16]


def fetch_rest(
    source: LiveSourceConfig,
    *,
    query: str,
    params: dict[str, str] | None = None,
    timeout_sec: int | None = None,
    max_items: int = 20,
    max_body_bytes: int = 2_000_000,
) -> tuple[list[LiveItem], str]:
    params = params or {}
    _apply_rate_limit(source.name, source.rate_limit_rpm)
    endpoint = source.endpoint or source.url
    if not endpoint:
        return [], ""
    try:
        endpoint = endpoint.format(**params)
    except Exception:
        pass
    q = dict(source.query_params)
    q.update(params)
    if query and "q" not in q and "query" not in q:
        q["q"] = query
    headers = _headers_from_auth(source.auth)
    method = source.method.upper()
    req_timeout = int(timeout_sec or source.timeout_sec or 20)
    if method == "POST":
        r = requests.post(endpoint, json=q, headers=headers, timeout=req_timeout)
    else:
        r = requests.get(endpoint, params=q, headers=headers, timeout=req_timeout)
    r.raise_for_status()
    raw = r.text or ""
    if len(raw.encode("utf-8")) > max_body_bytes:
        raw = raw[:max_body_bytes]
    payload = r.json() if raw else {}

    transform = source.transform or {}
    mode = str(transform.get("mode") or "template").strip().lower()
    item_path = str(transform.get("item_path") or "").strip()
    tpl = str(transform.get("snippet_template") or "{value}")
    rows: list[LiveItem] = []
    if mode == "template" and item_path:
        value = _path_get(payload, item_path)
        values = value if isinstance(value, list) else [value]
        for v in values[: max(1, int(max_items))]:
            txt = tpl.format(value=v)
            rows.append(
                LiveItem(
                    id=_id_for(source.name, txt, endpoint),
                    url=endpoint,
                    title=None,
                    text=str(txt),
                    published_at=datetime.now(timezone.utc).isoformat(),
                    source=source.name,
                    raw_meta={"value": v},
                )
            )
    else:
        txt = json.dumps(payload, ensure_ascii=True)[:2000]
        rows.append(
            LiveItem(
                id=_id_for(source.name, txt, endpoint),
                url=endpoint,
                title=None,
                text=txt,
                published_at=datetime.now(timezone.utc).isoformat(),
                source=source.name,
                raw_meta={},
            )
        )
    return rows, raw


def fetch_rss(
    source: LiveSourceConfig,
    *,
    query: str,
    timeout_sec: int | None = None,
    max_items: int = 20,
    max_body_bytes: int = 2_000_000,
) -> tuple[list[LiveItem], str]:
    _apply_rate_limit(source.name, source.rate_limit_rpm)
    url = source.url or source.endpoint
    if not url:
        return [], ""
    r = requests.get(url, timeout=int(timeout_sec or source.timeout_sec or 20))
    r.raise_for_status()
    raw = r.text or ""
    if len(raw.encode("utf-8")) > max_body_bytes:
        raw = raw[:max_body_bytes]
    root = ET.fromstring(raw)
    rows: list[LiveItem] = []
    items = root.findall(".//item")
    if not items:
        items = root.findall(".//{http://www.w3.org/2005/Atom}entry")
    for item in items[: max(1, int(max_items))]:
        title = (item.findtext("title") or item.findtext("{http://www.w3.org/2005/Atom}title") or "").strip()
        link = (item.findtext("link") or "").strip()
        if not link:
            atom_link = item.find("{http://www.w3.org/2005/Atom}link")
            if atom_link is not None:
                link = str(atom_link.attrib.get("href") or "").strip()
        desc = (
            item.findtext("description")
            or item.findtext("summary")
            or item.findtext("{http://www.w3.org/2005/Atom}summary")
            or ""
        ).strip()
        pub = (
            item.findtext("pubDate")
            or item.findtext("published")
            or item.findtext("{http://www.w3.org/2005/Atom}published")
            or ""
        ).strip()
        text = f"{title}: {desc}".strip(": ").strip()
        if query and query.lower() not in text.lower():
            # permissive filter: include if query is empty or appears in title/summary
            pass
        rows.append(
            LiveItem(
                id=_id_for(source.name, text, link or url),
                url=link or url,
                title=title or None,
                text=text or title or desc,
                published_at=pub or None,
                source=source.name,
                raw_meta={},
            )
        )
    return rows, raw


def fetch_scrape(
    source: LiveSourceConfig,
    *,
    query: str,
    params: dict[str, str] | None = None,
    timeout_sec: int | None = None,
    max_items: int = 20,
    max_body_bytes: int = 2_000_000,
) -> tuple[list[LiveItem], str]:
    params = params or {}
    _apply_rate_limit(source.name, source.rate_limit_rpm)
    url = source.url or source.endpoint
    if not url:
        return [], ""
    try:
        url = url.format(**params)
    except Exception:
        pass
    r = requests.get(url, timeout=int(timeout_sec or source.timeout_sec or 20))
    r.raise_for_status()
    raw = r.text or ""
    if len(raw.encode("utf-8")) > max_body_bytes:
        raw = raw[:max_body_bytes]
    text = re.sub(r"(?is)<script.*?>.*?</script>", " ", raw)
    text = re.sub(r"(?is)<style.*?>.*?</style>", " ", text)
    text = re.sub(r"(?s)<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    if query and query.lower() not in text.lower():
        text = text[:3000]
    rows = [
        LiveItem(
            id=_id_for(source.name, text, url),
            url=url,
            title=None,
            text=text[:4000],
            published_at=datetime.now(timezone.utc).isoformat(),
            source=source.name,
            raw_meta={},
        )
    ]
    return rows[: max(1, int(max_items))], raw


def fetch_live_source(
    source: LiveSourceConfig,
    *,
    query: str,
    params: dict[str, str] | None = None,
    timeout_sec: int | None = None,
    max_items: int = 20,
    max_body_bytes: int = 2_000_000,
) -> tuple[list[LiveItem], str]:
    if source.type == "rest":
        return fetch_rest(
            source,
            query=query,
            params=params,
            timeout_sec=timeout_sec,
            max_items=max_items,
            max_body_bytes=max_body_bytes,
        )
    if source.type == "rss":
        return fetch_rss(
            source,
            query=query,
            timeout_sec=timeout_sec,
            max_items=max_items,
            max_body_bytes=max_body_bytes,
        )
    if source.type == "scrape":
        return fetch_scrape(
            source,
            query=query,
            params=params,
            timeout_sec=timeout_sec,
            max_items=max_items,
            max_body_bytes=max_body_bytes,
        )
    raise ValueError(f"unsupported_live_source_type:{source.type}")

