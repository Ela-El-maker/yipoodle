from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter, Request

from src.apps.query_router import load_router_config
from src.apps.sources_config import load_sources_config


router = APIRouter(prefix="/config", tags=["config"])


@router.get("/router")
def api_router_config(request: Request) -> dict[str, Any]:
    settings = request.app.state.ui_service.settings
    return {
        "path": settings.router_config,
        "config": load_router_config(settings.router_config),
    }


@router.get("/domains")
def api_domains(_request: Request) -> dict[str, Any]:
    root = Path("config/domains")
    rows: list[dict[str, Any]] = []
    if root.exists():
        for p in sorted(root.glob("sources_*.yaml")):
            rows.append({
                "name": p.stem.replace("sources_", ""),
                "path": str(p),
            })
    return {"count": len(rows), "domains": rows}


@router.get("/indexes")
def api_indexes(_request: Request) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for root in [Path("data/indexes"), Path("runs/monitor/data")]:
        if not root.exists():
            continue
        for p in sorted(root.rglob("*.json")):
            rows.append({"path": str(p), "name": p.name})
    return {"count": len(rows), "indexes": rows}


@router.get("/live-sources")
def api_live_sources(request: Request) -> dict[str, Any]:
    settings = request.app.state.ui_service.settings
    cfg = load_sources_config(settings.sources_config)
    live = cfg.get("live_sources", {}) if isinstance(cfg, dict) else {}
    out: list[dict[str, Any]] = []
    if isinstance(live, dict):
        for name, block in sorted(live.items()):
            if not isinstance(block, dict):
                continue
            out.append(
                {
                    "name": str(name),
                    "enabled": bool(block.get("enabled", False)),
                    "type": block.get("type"),
                    "domain_tags": list(block.get("domain_tags", []) or []),
                    "cache_ttl_sec": block.get("cache_ttl_sec"),
                    "timeout_sec": block.get("timeout_sec"),
                    "rate_limit_rpm": block.get("rate_limit_rpm"),
                }
            )
    return {"path": settings.sources_config, "count": len(out), "live_sources": out}
