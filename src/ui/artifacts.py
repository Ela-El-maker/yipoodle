from __future__ import annotations

from pathlib import Path
from typing import Any


ALLOWED_ARTIFACT_KEYS = {
    "output_path",
    "report_path",
    "evidence_path",
    "metrics_path",
    "report_json_path",
    "router_sidecar_path",
    "mode_sidecar_path",
    "monitor_spec_path",
    "generated_automation_config",
    "notes_report_path",
    "manifest_path",
}


def _resolve(path: str) -> Path:
    return Path(path).expanduser().resolve()


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def is_path_allowed(path: str, *, roots: list[str]) -> bool:
    p = _resolve(path)
    for r in roots:
        root = _resolve(r)
        if _is_within(p, root):
            return True
    return False


def resolve_artifact_path(*, run_details: dict[str, Any], artifact_key: str, roots: list[str]) -> Path:
    if artifact_key not in ALLOWED_ARTIFACT_KEYS:
        raise KeyError(f"artifact key not allowed: {artifact_key}")

    artifacts = run_details.get("artifacts") if isinstance(run_details, dict) else None
    raw: str | None = None
    if isinstance(artifacts, dict):
        got = artifacts.get(artifact_key)
        if isinstance(got, str):
            raw = got
    if raw is None:
        got = run_details.get(artifact_key) if isinstance(run_details, dict) else None
        if isinstance(got, str):
            raw = got
    if not raw:
        raise FileNotFoundError(f"artifact path not found for key: {artifact_key}")

    p = _resolve(raw)
    if not p.exists() or not p.is_file():
        raise FileNotFoundError(f"artifact file not found: {p}")
    if not is_path_allowed(str(p), roots=roots):
        raise PermissionError(f"artifact path outside allowed roots: {p}")
    return p
