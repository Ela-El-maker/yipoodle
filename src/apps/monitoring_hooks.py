from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, TimeoutError
import importlib
from typing import Any


_ALLOWED_SEVERITIES = {"low", "medium", "high"}


def _validate_hook_decision(raw: dict[str, Any], topic: str, run_id: str) -> dict[str, Any]:
    trigger_id = str(raw.get("trigger_id") or "custom_hook_trigger")
    severity = str(raw.get("severity") or "low").lower()
    if severity not in _ALLOWED_SEVERITIES:
        raise ValueError(f"invalid severity '{severity}'")
    return {
        "trigger_id": trigger_id,
        "type": "hook",
        "severity": severity,
        "fired": bool(raw.get("fired", False)),
        "reason": str(raw.get("reason", "")),
        "observed": raw.get("observed", {}),
        "threshold": raw.get("threshold", {}),
        "topic": topic,
        "run_id": run_id,
        "hook": str(raw.get("hook", "")),
    }


def run_monitor_hooks(
    *,
    hooks: list[dict[str, Any]],
    allowlist: list[str],
    timeout_sec: float,
    context: dict[str, Any],
    topic: str,
    run_id: str,
) -> tuple[list[dict[str, Any]], list[str]]:
    decisions: list[dict[str, Any]] = []
    errors: list[str] = []
    allowed = set(str(x) for x in (allowlist or []))

    for h in hooks or []:
        module_name = str(h.get("module") or "").strip()
        fn_name = str(h.get("function") or "evaluate_topic").strip()
        if not module_name:
            errors.append("hook module missing")
            continue
        if module_name not in allowed:
            errors.append(f"hook module not allowlisted: {module_name}")
            continue
        try:
            module = importlib.import_module(module_name)
            fn = getattr(module, fn_name)
        except Exception as exc:
            errors.append(f"hook import failed {module_name}.{fn_name}: {exc}")
            continue

        def _invoke() -> Any:
            return fn(context)

        try:
            with ThreadPoolExecutor(max_workers=1) as pool:
                fut = pool.submit(_invoke)
                raw_out = fut.result(timeout=float(timeout_sec))
        except TimeoutError:
            errors.append(f"hook timeout {module_name}.{fn_name}")
            continue
        except Exception as exc:
            errors.append(f"hook failed {module_name}.{fn_name}: {exc}")
            continue

        if not isinstance(raw_out, list):
            errors.append(f"hook output must be list: {module_name}.{fn_name}")
            continue

        for raw in raw_out:
            if not isinstance(raw, dict):
                errors.append(f"hook decision must be dict: {module_name}.{fn_name}")
                continue
            try:
                item = _validate_hook_decision(raw, topic, run_id)
                item["hook"] = f"{module_name}.{fn_name}"
                decisions.append(item)
            except Exception as exc:
                errors.append(f"hook decision invalid {module_name}.{fn_name}: {exc}")

    return decisions, errors
