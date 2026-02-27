from __future__ import annotations

from pathlib import Path
from typing import Any

import os
import yaml
from pydantic import BaseModel, Field


DEFAULT_UI_CONFIG: dict[str, Any] = {
    "host": "127.0.0.1",
    "port": 8080,
    "max_workers": 2,
    "run_db_path": "runs/ui/ui_runs.db",
    "artifacts_roots": [
        "runs/query",
        "runs/research_reports",
        "runs/monitor",
        "runs/notes",
        "runs/audit/runs",
    ],
    "sse_keepalive_sec": 10,
    "job_timeout_sec": {
        "ask": 30,
        "research": 1800,
        "notes": 2400,
        "monitor": 3600,
        "query": 2400,
        "automation": 7200,
    },
    "router_config": "config/router.yaml",
    "sources_config": "config/sources.yaml",
    "automation_config": "config/automation.yaml",
    "kb_db": "data/kb/knowledge.db",
    "vector_service_endpoint": None,
    "chat": {
        "enabled": True,
        "default_mode": "auto",
        "max_message_chars": 12000,
        "max_blob_bytes": 5_000_000,
        "sse_replay_limit": 500,
        "title_autogen": True,
        "retain_events_days": 30,
        "db_only_chat_outputs": True,
    },
}


class UISettings(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8080
    max_workers: int = 2
    run_db_path: str = "runs/ui/ui_runs.db"
    artifacts_roots: list[str] = Field(default_factory=lambda: list(DEFAULT_UI_CONFIG["artifacts_roots"]))
    sse_keepalive_sec: int = 10
    job_timeout_sec: dict[str, int] = Field(default_factory=lambda: dict(DEFAULT_UI_CONFIG["job_timeout_sec"]))
    router_config: str = "config/router.yaml"
    sources_config: str = "config/sources.yaml"
    automation_config: str = "config/automation.yaml"
    kb_db: str = "data/kb/knowledge.db"
    vector_service_endpoint: str | None = None
    chat_enabled: bool = True
    chat_default_mode: str = "auto"
    chat_max_message_chars: int = 12000
    chat_max_blob_bytes: int = 5_000_000
    chat_sse_replay_limit: int = 500
    chat_title_autogen: bool = True
    chat_retain_events_days: int = 30
    chat_db_only_outputs: bool = True


def _deep_merge(base: dict[str, Any], incoming: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in incoming.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _load_yaml(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    payload = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    return payload if isinstance(payload, dict) else {}


def load_ui_settings(config_path: str | None = "config/ui.yaml") -> UISettings:
    merged = dict(DEFAULT_UI_CONFIG)
    merged = _deep_merge(merged, _load_yaml(config_path))

    env_host = os.getenv("YIPOODLE_UI_HOST")
    env_port = os.getenv("YIPOODLE_UI_PORT")
    if env_host:
        merged["host"] = env_host
    if env_port:
        try:
            merged["port"] = int(env_port)
        except ValueError:
            pass

    chat_cfg = merged.get("chat", {}) if isinstance(merged.get("chat"), dict) else {}
    merged["chat_enabled"] = bool(chat_cfg.get("enabled", True))
    merged["chat_default_mode"] = str(chat_cfg.get("default_mode", "auto"))
    merged["chat_max_message_chars"] = int(chat_cfg.get("max_message_chars", 12000))
    merged["chat_max_blob_bytes"] = int(chat_cfg.get("max_blob_bytes", 5_000_000))
    merged["chat_sse_replay_limit"] = int(chat_cfg.get("sse_replay_limit", 500))
    merged["chat_title_autogen"] = bool(chat_cfg.get("title_autogen", True))
    merged["chat_retain_events_days"] = int(chat_cfg.get("retain_events_days", 30))
    merged["chat_db_only_outputs"] = bool(chat_cfg.get("db_only_chat_outputs", True))

    return UISettings.model_validate(merged)
