from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
import json

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from src.apps.automation import load_automation_config
from src.ui.chat_executors import UIChatExecutors
from src.ui.chat_queue import UIChatQueue
from src.ui.chat_store import UIChatStore
from src.ui.executors import UIExecutors
from src.ui.job_queue import UIJobQueue
from src.ui.routes.automation import router as automation_router
from src.ui.routes.chat import router as chat_router
from src.ui.routes.config import router as config_router
from src.ui.routes.health import router as health_router
from src.ui.routes.kb import router as kb_router
from src.ui.routes.monitor import router as monitor_router
from src.ui.routes.runs import router as runs_router
from src.ui.run_store import UIRunStore
from src.ui.settings import UISettings, load_ui_settings


class UIService:
    def __init__(self, settings: UISettings):
        self.settings = settings
        self.store = UIRunStore(settings.run_db_path)
        self.executors = UIExecutors(settings)
        self.job_queue = UIJobQueue(store=self.store, executors=self.executors, settings=settings)
        self.chat_store = UIChatStore(settings.run_db_path)
        self.chat_executors = UIChatExecutors(settings)
        self.chat_queue = UIChatQueue(store=self.chat_store, executors=self.chat_executors, settings=settings)


def _list_indexes() -> list[str]:
    rows: list[str] = []
    for root in [Path("data/indexes"), Path("runs/monitor/data")]:
        if not root.exists():
            continue
        for p in sorted(root.rglob("*.json")):
            rows.append(str(p))
    return rows


def _load_latest_summary(path: str) -> dict[str, Any] | None:
    p = Path(path)
    if not p.exists():
        return None
    try:
        payload = json.loads(p.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else None
    except Exception:
        return None


def create_app(config_path: str | None = "config/ui.yaml") -> FastAPI:
    settings = load_ui_settings(config_path)
    templates = Jinja2Templates(directory="src/ui/templates")
    service = UIService(settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        service.store.init()
        service.store.recover_stale_running()
        service.chat_store.init()
        service.chat_store.recover_stale_running()
        service.chat_store.prune_events_older_than(days=service.settings.chat_retain_events_days)
        await service.job_queue.start()
        await service.chat_queue.start()
        yield
        await service.job_queue.stop()
        await service.chat_queue.stop()

    app = FastAPI(title="Yipoodle Operator UI", version="1.0.0", lifespan=lifespan)
    app.state.ui_service = service
    app.mount("/static", StaticFiles(directory="src/ui/static"), name="static")

    api = FastAPI()
    api.state.ui_service = service
    api.include_router(health_router)
    api.include_router(config_router)
    api.include_router(runs_router)
    api.include_router(chat_router)
    api.include_router(automation_router)
    api.include_router(monitor_router)
    api.include_router(kb_router)
    app.mount("/api/v1", api)

    @app.get("/", response_class=HTMLResponse)
    def page_chat(request: Request):
        return templates.TemplateResponse(
            request,
            "chat.html",
            {
                "modes": ["auto", "ask", "research", "monitor", "notes", "query", "automation"],
                "default_mode": service.settings.chat_default_mode,
            },
        )

    @app.get("/run-console", response_class=HTMLResponse)
    def page_index(request: Request):
        return templates.TemplateResponse(
            request,
            "index.html",
            {
                "modes": ["ask", "research", "monitor", "notes", "query", "automation"],
                "indexes": _list_indexes(),
                "sources_config": service.settings.sources_config,
                "automation_config": service.settings.automation_config,
            },
        )

    @app.get("/runs", response_class=HTMLResponse)
    def page_runs(request: Request, run_id: str | None = None):
        runs = service.store.list_runs(limit=200)
        selected = None
        events: list[dict[str, Any]] = []
        if run_id:
            try:
                selected = service.store.get_run(run_id)
                events = service.store.list_events_since(run_id, after_seq=0, limit=500)
            except KeyError:
                selected = None
        return templates.TemplateResponse(
            request,
            "runs.html",
            {
                "runs": runs,
                "selected": selected,
                "events": events,
            },
        )

    @app.get("/monitor", response_class=HTMLResponse)
    def page_monitor(request: Request):
        topics: list[dict[str, Any]] = []
        topics_dir = Path("runs/monitor/topics")
        if topics_dir.exists():
            for p in sorted(topics_dir.glob("*.json")):
                try:
                    payload = json.loads(p.read_text(encoding="utf-8"))
                    if isinstance(payload, dict):
                        payload["_path"] = str(p)
                        topics.append(payload)
                except Exception:
                    topics.append({"name": p.stem, "_path": str(p), "_parse_error": True})

        latest_summary = None
        try:
            acfg = load_automation_config(service.settings.automation_config)
            audit_dir = str((acfg.get("paths", {}) or {}).get("audit_dir", "runs/audit"))
            latest_summary = _load_latest_summary(str(Path(audit_dir) / "latest_summary.json"))
        except Exception:
            latest_summary = None

        return templates.TemplateResponse(
            request,
            "monitor.html",
            {
                "topics": topics,
                "latest_summary": latest_summary,
                "automation_config": service.settings.automation_config,
            },
        )

    @app.get("/health", response_class=HTMLResponse)
    def page_health(request: Request):
        info = {
            "queue": service.job_queue.status(),
            "chat_queue": service.chat_queue.status(),
            "run_db_path": service.settings.run_db_path,
            "artifacts_roots": service.settings.artifacts_roots,
            "router_config": service.settings.router_config,
            "sources_config": service.settings.sources_config,
            "automation_config": service.settings.automation_config,
        }
        return templates.TemplateResponse(request, "health.html", {"info": info})

    return app
