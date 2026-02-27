from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
import json

from src.apps.query_router import RouterDecision, load_router_config, route_query
from src.ui.executors import UIExecutors
from src.ui.settings import UISettings


CHAT_SUPPORTED_MODES = {"auto", "ask", "query", "research", "notes", "monitor", "automation"}


@dataclass
class ChatBlob:
    blob_type: str
    blob_text: str


@dataclass
class ChatExecuteResult:
    mode_used: str
    content_markdown: str
    metadata: dict[str, Any]
    blobs: list[ChatBlob]


def _coerce_mode(mode: str | None) -> str:
    m = str(mode or "auto").strip().lower()
    return m if m in CHAT_SUPPORTED_MODES else "auto"


def _chunk_text(text: str, size: int = 220) -> list[str]:
    payload = text or ""
    if not payload:
        return []
    return [payload[i : i + size] for i in range(0, len(payload), size)]


def _read_text(path: str) -> str | None:
    p = Path(path)
    if not p.exists() or not p.is_file():
        return None
    return p.read_text(encoding="utf-8", errors="replace")


def _format_json_markdown(data: dict[str, Any]) -> str:
    return "```json\n" + json.dumps(data, indent=2, ensure_ascii=True) + "\n```\n"


def _extract_primary_markdown(mode: str, details: dict[str, Any], output_path: str | None) -> str:
    preferred = [
        details.get("notes_path"),
        details.get("report_path"),
        output_path,
    ]
    for raw in preferred:
        if not isinstance(raw, str):
            continue
        if raw.lower().endswith(".md"):
            content = _read_text(raw)
            if content:
                return content
    return _format_json_markdown(details)


def _blob_type_for_artifact(mode: str, key: str, path: str) -> str:
    suffix = Path(path).suffix.lower()
    if key == "report_path" or suffix == ".md":
        return "report_md"
    if key == "evidence_path":
        return "evidence_json"
    if key == "metrics_path":
        return "metrics_json"
    if key == "router_sidecar_path":
        return "router_json"
    if key in {"monitor_spec_path", "generated_automation_config"}:
        return "monitor_json"
    if key == "manifest_path":
        return "automation_summary_json"
    if key == "mode_sidecar_path":
        if mode == "notes":
            return "kb_json"
        if mode == "monitor":
            return "monitor_json"
        if mode == "automation":
            return "automation_summary_json"
        return "router_json"
    if suffix == ".json":
        return "monitor_json"
    return "report_md"


class UIChatExecutors:
    def __init__(self, settings: UISettings):
        self.settings = settings
        self._run_exec = UIExecutors(settings)

    def execute(
        self,
        *,
        message_id: str,
        request: dict[str, Any],
        emit_event: Callable[[str, str, dict[str, Any]], None],
        cancel_requested: Callable[[], bool],
    ) -> ChatExecuteResult:
        prompt = str(request.get("content") or "").strip()
        if not prompt and _coerce_mode(request.get("mode")) != "automation":
            raise ValueError("content is required for this mode")

        requested_mode = _coerce_mode(request.get("mode"))
        router_cfg = load_router_config(self.settings.router_config)
        decision: RouterDecision | None = None
        mode_used = requested_mode
        if requested_mode == "auto":
            decision = route_query(prompt, router_cfg, explicit_mode="auto")
            mode_used = decision.mode
            emit_event("info", "routing", {"mode_used": mode_used, "reason": decision.reason, "signals": decision.signals})

        run_request = self._to_run_request(mode=mode_used, prompt=prompt, request=request)
        emit_event("info", "dispatch", {"mode_used": mode_used})

        def _adapt_emit(level: str, message: str, payload: dict[str, Any]) -> None:
            emit_event(level, "progress", {"message": message, **(payload or {})})

        result = self._run_exec.execute(
            run_id=message_id,
            request=run_request,
            emit_event=_adapt_emit,
            cancel_requested=cancel_requested,
        )

        details = dict(result.details)
        content_markdown = _extract_primary_markdown(mode_used, details, result.output_path)
        blobs, blob_meta = self._collect_blobs(mode=mode_used, details=details)

        metadata: dict[str, Any] = {
            "mode_requested": requested_mode,
            "mode_used": mode_used,
            "details": details,
            "blob_meta": blob_meta,
        }
        if decision is not None:
            metadata["routing"] = {
                "mode": decision.mode,
                "confidence": decision.confidence,
                "reason": decision.reason,
                "signals": decision.signals,
                "override_used": decision.override_used,
            }

        token_chunks = _chunk_text(content_markdown)
        for idx, chunk in enumerate(token_chunks):
            emit_event("info", "token", {"index": idx, "delta": chunk})

        return ChatExecuteResult(
            mode_used=mode_used,
            content_markdown=content_markdown,
            metadata=metadata,
            blobs=blobs,
        )

    def _to_run_request(self, *, mode: str, prompt: str, request: dict[str, Any]) -> dict[str, Any]:
        options = dict(request.get("options") or {})
        return {
            "mode": mode,
            "question": prompt if mode != "automation" else options.get("question"),
            "index": options.get("index"),
            "sources_config": str(options.get("sources_config") or self.settings.sources_config),
            "automation_config": str(options.get("automation_config") or self.settings.automation_config),
            "output_path": options.get("output_path"),
            "options": options,
        }

    def _collect_blobs(self, *, mode: str, details: dict[str, Any]) -> tuple[list[ChatBlob], dict[str, Any]]:
        blobs: list[ChatBlob] = []
        meta: dict[str, Any] = {"truncated": [], "added": []}
        max_bytes = int(self.settings.chat_max_blob_bytes)
        artifacts = details.get("artifacts") if isinstance(details.get("artifacts"), dict) else {}
        candidates: dict[str, str] = {}
        if isinstance(artifacts, dict):
            for k, v in artifacts.items():
                if isinstance(v, str):
                    candidates[k] = v
        for key in [
            "report_path",
            "evidence_path",
            "metrics_path",
            "router_sidecar_path",
            "mode_sidecar_path",
            "monitor_spec_path",
            "manifest_path",
            "notes_report_path",
        ]:
            raw = details.get(key)
            if isinstance(raw, str):
                candidates.setdefault(key, raw)

        seen: set[tuple[str, str]] = set()
        for key, path in candidates.items():
            p = Path(path)
            if not p.exists() or not p.is_file():
                continue
            blob_type = _blob_type_for_artifact(mode, key, path)
            raw = p.read_text(encoding="utf-8", errors="replace")
            enc = raw.encode("utf-8", errors="replace")
            if len(enc) > max_bytes:
                trimmed = enc[:max_bytes].decode("utf-8", errors="ignore")
                raw = trimmed + "\n\n[TRUNCATED]"
                meta["truncated"].append({"blob_type": blob_type, "path": str(p), "max_bytes": max_bytes})
            sig = (blob_type, raw)
            if sig in seen:
                continue
            seen.add(sig)
            blobs.append(ChatBlob(blob_type=blob_type, blob_text=raw))
            meta["added"].append({"blob_type": blob_type, "path": str(p)})
        return blobs, meta
