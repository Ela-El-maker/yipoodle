from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import platform
import shutil
import subprocess
import sys


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _safe_cmd(args: list[str]) -> str:
    try:
        return subprocess.check_output(args, text=True, stderr=subprocess.STDOUT).strip()
    except Exception as exc:
        return f"unavailable: {exc}"


def create_snapshot(
    out_dir: str,
    report_path: str | None = None,
    index_path: str | None = None,
    config_paths: list[str] | None = None,
) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    root = Path(out_dir) / f"snapshot_{ts}"
    artifacts = root / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)

    copied: list[dict[str, str]] = []

    def copy_if_exists(path_str: str) -> None:
        p = Path(path_str)
        if not p.exists():
            return
        dest = artifacts / p.name
        shutil.copy2(p, dest)
        copied.append({"src": str(p), "dst": str(dest), "sha256": _sha256(dest)})

    if report_path:
        report = Path(report_path)
        copy_if_exists(str(report))
        copy_if_exists(str(report.with_suffix(".json")))
        copy_if_exists(str(report.with_suffix(".evidence.json")))
        copy_if_exists(str(report.with_suffix(".metrics.json")))

    if index_path:
        copy_if_exists(index_path)

    for cfg in config_paths or []:
        copy_if_exists(cfg)

    copy_if_exists("requirements.txt")
    copy_if_exists("docs/checklist.md")
    copy_if_exists("README.md")

    manifest = {
        "created_utc": ts,
        "python": sys.version,
        "platform": platform.platform(),
        "executable": sys.executable,
        "pip_freeze": _safe_cmd([sys.executable, "-m", "pip", "freeze"]),
        "git_head": _safe_cmd(["git", "rev-parse", "HEAD"]),
        "git_status": _safe_cmd(["git", "status", "--short"]),
        "artifacts": copied,
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return str(root)
