from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
import json
import time

from src.apps.ask_mode import run_ask_mode
from src.apps.automation import run_automation
from src.apps.monitor_mode import run_monitor_mode
from src.apps.notes_mode import run_notes_mode
from src.apps.query_router import dispatch_query, load_router_config, route_query
from src.apps.research_copilot import run_research
from src.ui.settings import UISettings


@dataclass
class ExecuteResult:
    output_path: str | None
    details: dict[str, Any]


def _slug(text: str) -> str:
    out = "".join(ch.lower() if ch.isalnum() else "_" for ch in (text or "").strip())
    out = "_".join(part for part in out.split("_") if part)
    return out or "query"


def _utc_ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _ensure_parent(path: str) -> str:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return str(p)


def _artifact_map(out_path: str | None, mode: str, details: dict[str, Any]) -> dict[str, str]:
    artifacts: dict[str, str] = {}
    if out_path:
        artifacts["output_path"] = str(Path(out_path))
        p = Path(out_path)
        if p.suffix.lower() == ".md":
            report_json = p.with_suffix(".json")
            evidence_json = p.with_suffix(".evidence.json")
            metrics_json = p.with_suffix(".metrics.json")
            if report_json.exists():
                artifacts["report_json_path"] = str(report_json)
                artifacts["mode_sidecar_path"] = str(report_json)
            if evidence_json.exists():
                artifacts["evidence_path"] = str(evidence_json)
            if metrics_json.exists():
                artifacts["metrics_path"] = str(metrics_json)
            if p.exists():
                artifacts["report_path"] = str(p)
        elif p.exists():
            artifacts["mode_sidecar_path"] = str(p)
    if mode == "query" and out_path:
        router_sidecar = str(Path(out_path).with_suffix(".router.json"))
        if Path(router_sidecar).exists():
            artifacts["router_sidecar_path"] = router_sidecar

    for key in [
        "monitor_spec_path",
        "generated_automation_config",
        "baseline_run_dir",
        "manifest_path",
        "automation_run_dir",
        "router_sidecar_path",
        "mode_sidecar_path",
        "notes_report_path",
    ]:
        val = details.get(key)
        if isinstance(val, str) and val.strip():
            artifacts[key] = val

    # run directories are not file artifacts but useful metadata.
    return artifacts


class UIExecutors:
    def __init__(self, settings: UISettings):
        self.settings = settings

    def execute(
        self,
        *,
        run_id: str,
        request: dict[str, Any],
        emit_event: Callable[[str, str, dict[str, Any]], None],
        cancel_requested: Callable[[], bool],
    ) -> ExecuteResult:
        mode = str(request.get("mode") or "").strip().lower()
        if mode not in {"ask", "research", "monitor", "notes", "query", "automation"}:
            raise ValueError(f"unsupported mode: {mode}")

        emit_event("info", "execute_mode", {"mode": mode})
        if cancel_requested():
            raise RuntimeError("cancelled_before_start")

        if mode == "ask":
            result = self._exec_ask(run_id=run_id, request=request, emit_event=emit_event)
        elif mode == "research":
            result = self._exec_research(run_id=run_id, request=request, emit_event=emit_event)
        elif mode == "notes":
            result = self._exec_notes(run_id=run_id, request=request, emit_event=emit_event)
        elif mode == "monitor":
            result = self._exec_monitor(run_id=run_id, request=request, emit_event=emit_event)
        elif mode == "query":
            result = self._exec_query(run_id=run_id, request=request, emit_event=emit_event)
        else:
            result = self._exec_automation(run_id=run_id, request=request, emit_event=emit_event)

        details = dict(result.details)
        details["artifacts"] = _artifact_map(result.output_path, mode, details)
        return ExecuteResult(output_path=result.output_path, details=details)

    def _exec_ask(self, *, run_id: str, request: dict[str, Any], emit_event: Callable[[str, str, dict[str, Any]], None]) -> ExecuteResult:
        question = str(request.get("question") or "").strip()
        if not question:
            raise ValueError("ask mode requires question")
        out = str(request.get("output_path") or Path("runs/query") / run_id / f"ask_{_slug(question)[:80]}.md")
        router_cfg = load_router_config(str(request.get("options", {}).get("router_config") or self.settings.router_config))
        glossary = str(request.get("options", {}).get("glossary") or "config/ask_glossary.yaml")
        out = _ensure_parent(out)
        emit_event("info", "ask_started", {"out": out})
        payload = run_ask_mode(
            question=question,
            out_path=out,
            router_cfg=router_cfg,
            glossary_path=glossary,
        )
        emit_event("info", "ask_finished", {"out": payload.get("out_path")})
        return ExecuteResult(output_path=payload.get("out_path"), details=payload)

    def _research_kwargs(self, request: dict[str, Any], out: str) -> dict[str, Any]:
        options = dict(request.get("options") or {})
        kwargs: dict[str, Any] = {
            "index_path": str(request.get("index") or "data/indexes/bm25_index.json"),
            "question": str(request.get("question") or "").strip(),
            "top_k": int(options.get("top_k", 8)),
            "out_path": out,
            "min_items": int(options.get("min_items", 2)),
            "min_score": float(options.get("min_score", 0.5)),
            "retrieval_mode": str(options.get("retrieval_mode", "lexical")),
            "alpha": float(options.get("alpha", 0.6)),
            "max_per_paper": int(options.get("max_per_paper", 2)),
            "vector_index_path": options.get("vector_index_path"),
            "vector_metadata_path": options.get("vector_metadata_path"),
            "embedding_model": str(options.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2")),
            "quality_prior_weight": float(options.get("quality_prior_weight", 0.15)),
            "sources_config_path": str(request.get("sources_config") or self.settings.sources_config),
            "semantic_mode": str(options.get("semantic_mode", "offline")),
            "semantic_model": str(options.get("semantic_model", "sentence-transformers/all-MiniLM-L6-v2")),
            "semantic_min_support": float(options.get("semantic_min_support", 0.55)),
            "semantic_max_contradiction": float(options.get("semantic_max_contradiction", 0.30)),
            "semantic_shadow_mode": bool(options.get("semantic_shadow_mode", True)),
            "semantic_fail_on_low_support": bool(options.get("semantic_fail_on_low_support", False)),
            "online_semantic_model": str(options.get("online_semantic_model", "gpt-4o-mini")),
            "online_semantic_timeout_sec": float(options.get("online_semantic_timeout_sec", 12.0)),
            "online_semantic_max_checks": int(options.get("online_semantic_max_checks", 12)),
            "online_semantic_on_warn_only": bool(options.get("online_semantic_on_warn_only", True)),
            "online_semantic_base_url": options.get("online_semantic_base_url"),
            "online_semantic_api_key": options.get("online_semantic_api_key"),
            "vector_service_endpoint": options.get("vector_service_endpoint") or self.settings.vector_service_endpoint,
            "vector_nprobe": int(options.get("vector_nprobe", 16)),
            "vector_ef_search": int(options.get("vector_ef_search", 64)),
            "vector_topk_candidate_multiplier": float(options.get("vector_topk_candidate_multiplier", 1.5)),
            "live_enabled": bool(options.get("live", False)),
            "live_sources_override": options.get("live_sources"),
            "live_max_items": int(options.get("live_max_items", 20)),
            "live_timeout_sec": int(options.get("live_timeout_sec", 20)),
            "live_cache_ttl_sec": options.get("live_cache_ttl_sec"),
            "live_merge_mode": str(options.get("live_merge_mode", "union")),
            "routing_mode": str(options.get("routing_mode", "auto")),
            "intent": options.get("intent"),
            "relevance_policy": str(options.get("relevance_policy", "not_found")),
            "diagnostics": bool(options.get("diagnostics", True)),
            "direct_answer_mode": str(options.get("direct_answer_mode", "hybrid")),
            "direct_answer_max_complexity": int(options.get("direct_answer_max_complexity", 2)),
            "use_kb": bool(options.get("use_kb", False)),
            "kb_db": str(options.get("kb_db", self.settings.kb_db)),
            "kb_top_k": int(options.get("kb_top_k", 5)),
            "kb_merge_weight": float(options.get("kb_merge_weight", 0.15)),
            "use_cache": bool(options.get("use_cache", True)),
            "cache_dir": str(options.get("cache_dir", "runs/cache/research")),
            "aggregate": bool(options.get("aggregate", False)),
            "aggregate_model": str(options.get("aggregate_model", "sentence-transformers/all-MiniLM-L6-v2")),
            "aggregate_similarity_threshold": float(options.get("aggregate_similarity_threshold", 0.55)),
            "aggregate_contradiction_threshold": float(options.get("aggregate_contradiction_threshold", 0.40)),
        }
        return kwargs

    def _exec_research(
        self,
        *,
        run_id: str,
        request: dict[str, Any],
        emit_event: Callable[[str, str, dict[str, Any]], None],
        forced_out: str | None = None,
    ) -> ExecuteResult:
        question = str(request.get("question") or "").strip()
        if not question:
            raise ValueError("research mode requires question")
        out = forced_out or str(
            request.get("output_path")
            or (Path("runs/research_reports/ui") / f"{run_id}_research_{_slug(question)[:80]}.md")
        )
        out = _ensure_parent(out)
        kwargs = self._research_kwargs(request, out)
        emit_event("info", "research_started", {"out": out})
        out_path = run_research(**kwargs)
        emit_event("info", "research_finished", {"out": out_path})
        return ExecuteResult(
            output_path=out_path,
            details={
                "mode": "research",
                "report_path": out_path,
                "evidence_path": str(Path(out_path).with_suffix(".evidence.json")),
                "metrics_path": str(Path(out_path).with_suffix(".metrics.json")),
                "mode_sidecar_path": str(Path(out_path).with_suffix(".json")),
            },
        )

    def _exec_notes(self, *, run_id: str, request: dict[str, Any], emit_event: Callable[[str, str, dict[str, Any]], None]) -> ExecuteResult:
        question = str(request.get("question") or "").strip()
        if not question:
            raise ValueError("notes mode requires question")
        topic = request.get("options", {}).get("topic")
        out = str(
            request.get("output_path")
            or Path("runs/notes") / f"{run_id}_{_slug(str(topic or question))[:80]}.md"
        )
        out = _ensure_parent(out)
        emit_event("info", "notes_started", {"out": out})
        payload = run_notes_mode(
            question=question,
            index_path=str(request.get("index") or "data/indexes/bm25_index.json"),
            kb_db=str(request.get("options", {}).get("kb_db") or self.settings.kb_db),
            topic=str(topic) if topic else None,
            out_path=out,
            sources_config_path=str(request.get("sources_config") or self.settings.sources_config),
        )
        emit_event("info", "notes_finished", {"out": payload.get("notes_path")})
        return ExecuteResult(output_path=payload.get("notes_path"), details=payload)

    def _exec_monitor(self, *, run_id: str, request: dict[str, Any], emit_event: Callable[[str, str, dict[str, Any]], None]) -> ExecuteResult:
        question = str(request.get("question") or "").strip()
        if not question:
            raise ValueError("monitor mode requires question")
        opts = dict(request.get("options") or {})
        schedule = str(opts.get("schedule") or "0 */6 * * *")
        out = str(request.get("output_path") or Path("runs/monitor") / f"{_slug(question)[:80]}.json")
        out = _ensure_parent(out)
        emit_event("info", "monitor_started", {"out": out, "schedule": schedule})
        payload = run_monitor_mode(
            question=question,
            schedule=schedule,
            automation_config_path=str(request.get("automation_config") or self.settings.automation_config),
            out_path=out,
            register_schedule=bool(opts.get("register_schedule", True)),
            schedule_backend=str(opts.get("schedule_backend", "auto")),
        )
        emit_event("info", "monitor_finished", {"name": payload.get("name")})
        if payload.get("baseline_run_dir"):
            payload["automation_run_dir"] = payload.get("baseline_run_dir")
            payload["manifest_path"] = str(Path(str(payload.get("baseline_run_dir"))) / "manifest.json")
        return ExecuteResult(output_path=out, details=payload)

    def _exec_automation(self, *, run_id: str, request: dict[str, Any], emit_event: Callable[[str, str, dict[str, Any]], None]) -> ExecuteResult:
        cfg_path = str(request.get("automation_config") or self.settings.automation_config)
        emit_event("info", "automation_started", {"config": cfg_path})
        run_dir = run_automation(cfg_path)
        manifest_path = str(Path(run_dir) / "manifest.json")
        details = {
            "mode": "automation",
            "automation_config": cfg_path,
            "automation_run_dir": run_dir,
            "manifest_path": manifest_path,
        }
        emit_event("info", "automation_finished", {"run_dir": run_dir})
        return ExecuteResult(output_path=manifest_path, details=details)

    def _exec_query(self, *, run_id: str, request: dict[str, Any], emit_event: Callable[[str, str, dict[str, Any]], None]) -> ExecuteResult:
        question = str(request.get("question") or "").strip()
        if not question:
            raise ValueError("query mode requires question")
        options = dict(request.get("options") or {})
        router_cfg = load_router_config(str(options.get("router_config") or self.settings.router_config))
        explicit_mode = str(options.get("mode", "auto"))
        decision = route_query(question, router_cfg, explicit_mode=explicit_mode)
        out = str(
            request.get("output_path")
            or (Path("runs/query") / run_id / f"{decision.mode}_{_slug(question)[:80]}.md")
        )
        out = _ensure_parent(out)
        dispatch_started_utc = datetime.now(timezone.utc).isoformat()
        t0 = time.perf_counter()

        def _run_ask() -> dict[str, Any]:
            payload = run_ask_mode(
                question=question,
                out_path=out,
                router_cfg=router_cfg,
                glossary_path=str(options.get("glossary") or "config/ask_glossary.yaml"),
            )
            return {"output_path": payload["out_path"], "details": payload}

        def _run_research() -> dict[str, Any]:
            exec_res = self._exec_research(run_id=run_id, request=request, emit_event=emit_event, forced_out=out)
            return {"output_path": exec_res.output_path, "details": exec_res.details}

        def _run_monitor() -> dict[str, Any]:
            mon_req = dict(request)
            mon_opts = dict(options)
            if "schedule" in options:
                mon_opts["schedule"] = options["schedule"]
            mon_req["options"] = mon_opts
            exec_res = self._exec_monitor(run_id=run_id, request=mon_req, emit_event=emit_event)
            md = [
                "# Monitor",
                "",
                f"- question: {question}",
                f"- query: {exec_res.details.get('query')}",
                f"- schedule: {exec_res.details.get('schedule')}",
                f"- monitor_spec_path: {exec_res.details.get('monitor_spec_path')}",
                f"- generated_automation_config: {exec_res.details.get('generated_automation_config')}",
                f"- monitor_bootstrap_ok: {exec_res.details.get('monitor_bootstrap_ok')}",
                f"- baseline_run_id: {exec_res.details.get('baseline_run_id')}",
                f"- baseline_run_dir: {exec_res.details.get('baseline_run_dir')}",
            ]
            Path(out).write_text("\n".join(md) + "\n", encoding="utf-8")
            monitor_json = str(Path(out).with_suffix(".monitor.json"))
            Path(monitor_json).write_text(json.dumps(exec_res.details, indent=2), encoding="utf-8")
            details = dict(exec_res.details)
            details["mode_sidecar_path"] = monitor_json
            return {"output_path": out, "details": details}

        def _run_notes() -> dict[str, Any]:
            notes_req = dict(request)
            notes_req["output_path"] = out
            exec_res = self._exec_notes(run_id=run_id, request=notes_req, emit_event=emit_event)
            return {"output_path": exec_res.output_path, "details": exec_res.details}

        emit_event("info", "query_routed", {"selected_mode": decision.mode, "reason": decision.reason})
        dispatch = dispatch_query(
            decision,
            {
                "ask": _run_ask,
                "research": _run_research,
                "monitor": _run_monitor,
                "notes": _run_notes,
            },
        )

        elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 3)
        router_sidecar = str(Path(out).with_suffix(".router.json"))
        router_payload = {
            "question": question,
            "mode_selected": decision.mode,
            "mode_confidence": decision.confidence,
            "mode_reason": decision.reason,
            "signals": decision.signals,
            "override_used": decision.override_used,
            "dispatch_started_utc": dispatch_started_utc,
            "dispatch_elapsed_ms": elapsed_ms,
            "output_path": dispatch["output_path"],
        }
        Path(router_sidecar).write_text(json.dumps(router_payload, indent=2), encoding="utf-8")

        details = dict(dispatch.get("details") or {})
        details.update({
            "router_sidecar_path": router_sidecar,
            "mode_selected": decision.mode,
            "mode_reason": decision.reason,
            "mode_confidence": decision.confidence,
            "override_used": decision.override_used,
        })
        return ExecuteResult(output_path=str(dispatch["output_path"]), details=details)
