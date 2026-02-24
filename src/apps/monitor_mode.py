from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import json
import re
import shlex
import subprocess
import sys

import yaml

from src.apps.automation import load_automation_config, run_automation


def _utc_ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _slug(text: str) -> str:
    out = "".join(ch.lower() if ch.isalnum() else "_" for ch in text.strip())
    out = "_".join(part for part in out.split("_") if part)
    return out or "monitor_topic"


def _monitor_query_from_prompt(question: str) -> str:
    q = question.strip()
    q = re.sub(r"^(monitor|track|notify\s+me\s+if|alert\s+me\s+if)\b[:\-\s]*", "", q, flags=re.IGNORECASE).strip()
    return q or question.strip()


def _build_monitor_topic_cfg(name: str, query: str, schedule: str) -> dict[str, Any]:
    return {
        "name": name,
        "query": query,
        "max_results": 20,
        "top_k": 8,
        "min_items": 2,
        "min_score": 0.1,
        "monitoring_enabled": True,
        "monitoring": {
            "schedule": schedule,
            "cooldown_minutes": 360,
            "hysteresis_runs": 2,
            "triggers": [
                {"id": "new_docs", "type": "new_documents", "severity": "medium", "min_new_sources": 1},
                {"id": "claim_shift", "type": "kb_diff_count", "severity": "high", "min_added_claims": 1},
            ],
        },
        "kb_enabled": True,
        "kb_topic": name,
        "kb_diff_alert": True,
    }


def _root_dir() -> Path:
    return Path(__file__).resolve().parents[2]


def _python_exec() -> str:
    root = _root_dir()
    venv_py = root / ".venv" / "bin" / "python"
    if venv_py.exists():
        return str(venv_py)
    return sys.executable


def _monitor_cron_line(*, name: str, schedule: str, generated_config_path: str) -> str:
    marker = f"# yipoodle-monitor:{name}"
    root = _root_dir()
    log_path = root / "runs" / "monitor" / "audit" / name / "cron.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    py = shlex.quote(_python_exec())
    auto_py = shlex.quote(str(root / "scripts" / "auto_update.py"))
    cfg = shlex.quote(str(Path(generated_config_path).resolve()))
    log = shlex.quote(str(log_path))
    cmd = f"cd {shlex.quote(str(root))} && {py} {auto_py} --config {cfg} >> {log} 2>&1"
    return f"{schedule} /bin/bash -lc '{cmd}' {marker}"


def _read_crontab() -> str:
    proc = subprocess.run(["crontab", "-l"], capture_output=True, text=True)
    if proc.returncode == 0:
        return proc.stdout
    # Treat missing crontab as empty.
    stderr = (proc.stderr or "").lower()
    if "no crontab" in stderr or proc.returncode == 1:
        return ""
    raise RuntimeError(f"failed to read crontab: {proc.stderr.strip() or proc.returncode}")


def _write_crontab(text: str) -> None:
    proc = subprocess.run(["crontab", "-"], input=text, text=True, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(f"failed to write crontab: {proc.stderr.strip() or proc.returncode}")


def _upsert_monitor_crontab(*, name: str, schedule: str, generated_config_path: str) -> str:
    marker = f"# yipoodle-monitor:{name}"
    line = _monitor_cron_line(name=name, schedule=schedule, generated_config_path=generated_config_path)
    current = _read_crontab()
    kept: list[str] = []
    for raw in current.splitlines():
        if marker in raw:
            continue
        kept.append(raw)
    kept.append(line)
    out = "\n".join(kept).strip() + "\n"
    _write_crontab(out)
    return line


def _write_schedule_registry(
    *,
    name: str,
    schedule: str,
    generated_config_path: str,
    backend: str,
) -> str:
    path = Path("runs/monitor/schedules") / f"{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "name": name,
        "schedule": schedule,
        "generated_automation_config": str(generated_config_path),
        "backend": backend,
        "updated_utc": datetime.now(timezone.utc).isoformat(),
    }
    if path.exists():
        try:
            prev = json.loads(path.read_text(encoding="utf-8"))
            payload["created_utc"] = prev.get("created_utc", payload["updated_utc"])
        except Exception:
            payload["created_utc"] = payload["updated_utc"]
    else:
        payload["created_utc"] = payload["updated_utc"]
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return str(path)


def _remove_monitor_crontab(*, name: str) -> bool:
    marker = f"# yipoodle-monitor:{name}"
    current = _read_crontab()
    kept: list[str] = []
    removed = 0
    for raw in current.splitlines():
        if marker in raw:
            removed += 1
            continue
        kept.append(raw)
    if removed <= 0:
        return False
    out = ("\n".join(kept).strip() + "\n") if kept else ""
    _write_crontab(out)
    return True


def run_monitor_mode(
    *,
    question: str,
    schedule: str,
    automation_config_path: str,
    out_path: str,
    register_schedule: bool = True,
    schedule_backend: str = "auto",
) -> dict[str, Any]:
    ts = _utc_ts()
    cfg = load_automation_config(automation_config_path)
    query = _monitor_query_from_prompt(question)
    name = _slug(query)[:80]

    topics_dir = Path("runs/monitor/topics")
    gen_dir = Path("runs/monitor/generated")
    topics_dir.mkdir(parents=True, exist_ok=True)
    gen_dir.mkdir(parents=True, exist_ok=True)

    spec_path = topics_dir / f"{name}.json"
    spec = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "updated_utc": datetime.now(timezone.utc).isoformat(),
        "name": name,
        "question": question,
        "query": query,
        "schedule": schedule,
        "source_config": automation_config_path,
    }
    if spec_path.exists():
        prev = json.loads(spec_path.read_text(encoding="utf-8"))
        spec["created_utc"] = prev.get("created_utc", spec["created_utc"])
        spec["updated_utc"] = datetime.now(timezone.utc).isoformat()
    spec_path.write_text(json.dumps(spec, indent=2), encoding="utf-8")

    mon_cfg = dict(cfg)
    mon_cfg["topics"] = [_build_monitor_topic_cfg(name=name, query=query, schedule=schedule)]

    # Isolate monitor run paths to avoid clobbering base automation outputs.
    paths = dict(mon_cfg.get("paths", {}))
    paths["db_path"] = f"runs/monitor/data/{name}/papers.db"
    paths["papers_dir"] = f"runs/monitor/data/{name}/papers"
    paths["extracted_dir"] = f"runs/monitor/data/{name}/extracted"
    paths["index_path"] = f"runs/monitor/data/{name}/index.json"
    paths["reports_dir"] = f"runs/monitor/reports/{name}"
    paths["audit_dir"] = f"runs/monitor/audit/{name}"
    paths["cache_dir"] = f"runs/monitor/cache/{name}"
    mon_cfg["paths"] = paths

    gen_cfg_path = gen_dir / f"{name}.automation.yaml"
    gen_cfg_path.write_text(yaml.safe_dump(mon_cfg, sort_keys=False), encoding="utf-8")

    schedule_registered = False
    schedule_entry = None
    schedule_error = None
    schedule_backend_used = None
    backend_req = str(schedule_backend).strip().lower()
    if backend_req not in {"auto", "crontab", "file"}:
        raise ValueError("schedule_backend must be one of: auto, crontab, file")
    if register_schedule:
        if backend_req in {"auto", "crontab"}:
            try:
                schedule_entry = _upsert_monitor_crontab(
                    name=name,
                    schedule=schedule,
                    generated_config_path=str(gen_cfg_path),
                )
                schedule_registered = True
                schedule_backend_used = "crontab"
            except Exception as exc:
                schedule_error = str(exc)
                if backend_req == "crontab":
                    # explicit crontab backend requested: keep failure visible and do not fallback.
                    pass
        if not schedule_registered and backend_req in {"auto", "file"}:
            try:
                schedule_entry = _write_schedule_registry(
                    name=name,
                    schedule=schedule,
                    generated_config_path=str(gen_cfg_path),
                    backend="file",
                )
                schedule_registered = True
                schedule_backend_used = "file"
                if backend_req == "auto":
                    # fallback succeeded; keep original crontab error in payload for observability.
                    pass
            except Exception as exc:
                if schedule_error:
                    schedule_error = f"{schedule_error}; file_fallback_failed: {exc}"
                else:
                    schedule_error = str(exc)

    bootstrap_ok = True
    baseline_run_id = None
    baseline_run_dir = None
    bootstrap_error = None
    try:
        baseline_run_dir = run_automation(str(gen_cfg_path))
        baseline_run_id = Path(str(baseline_run_dir)).name
    except Exception as exc:
        bootstrap_ok = False
        bootstrap_error = str(exc)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "mode": "monitor",
        "name": name,
        "question": question,
        "query": query,
        "schedule": schedule,
        "monitor_spec_path": str(spec_path),
        "generated_automation_config": str(gen_cfg_path),
        "schedule_register_requested": bool(register_schedule),
        "schedule_backend_requested": backend_req,
        "schedule_backend_used": schedule_backend_used,
        "schedule_registered": schedule_registered,
        "schedule_entry": schedule_entry,
        "schedule_error": schedule_error,
        "monitor_bootstrap_ok": bootstrap_ok,
        "baseline_run_id": baseline_run_id,
        "baseline_run_dir": baseline_run_dir,
        "monitor_bootstrap_error": bootstrap_error,
        "created_at": ts,
    }
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def unregister_monitor(
    *,
    name_or_question: str,
    delete_files: bool = False,
) -> dict[str, Any]:
    name = _slug(name_or_question)[:80]
    spec_path = Path("runs/monitor/topics") / f"{name}.json"
    gen_cfg_path = Path("runs/monitor/generated") / f"{name}.automation.yaml"
    schedule_registry_path = Path("runs/monitor/schedules") / f"{name}.json"

    schedule_removed = False
    schedule_error = None
    try:
        schedule_removed = _remove_monitor_crontab(name=name)
    except Exception as exc:
        schedule_error = str(exc)

    spec_removed = False
    generated_config_removed = False
    schedule_registry_removed = False
    if delete_files:
        if spec_path.exists():
            spec_path.unlink()
            spec_removed = True
        if gen_cfg_path.exists():
            gen_cfg_path.unlink()
            generated_config_removed = True
        if schedule_registry_path.exists():
            schedule_registry_path.unlink()
            schedule_registry_removed = True

    return {
        "mode": "monitor_unreg",
        "name": name,
        "schedule_removed": schedule_removed,
        "schedule_error": schedule_error,
        "delete_files_requested": bool(delete_files),
        "spec_path": str(spec_path),
        "generated_automation_config": str(gen_cfg_path),
        "schedule_registry_path": str(schedule_registry_path),
        "spec_removed": spec_removed,
        "generated_config_removed": generated_config_removed,
        "schedule_registry_removed": schedule_registry_removed,
    }
