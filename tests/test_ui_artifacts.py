from __future__ import annotations

from pathlib import Path

from src.ui.artifacts import resolve_artifact_path


def test_artifact_resolution_and_whitelist(tmp_path: Path) -> None:
    root = tmp_path / "runs" / "query"
    root.mkdir(parents=True)
    f = root / "x.md"
    f.write_text("ok", encoding="utf-8")

    details = {
        "artifacts": {
            "report_path": str(f),
        }
    }
    p = resolve_artifact_path(run_details=details, artifact_key="report_path", roots=[str(tmp_path / "runs")])
    assert p == f.resolve()


def test_artifact_rejects_outside_roots(tmp_path: Path) -> None:
    outside = tmp_path / "outside.txt"
    outside.write_text("x", encoding="utf-8")
    details = {"artifacts": {"report_path": str(outside)}}
    try:
        resolve_artifact_path(run_details=details, artifact_key="report_path", roots=[str(tmp_path / "runs")])
    except PermissionError:
        pass
    else:
        raise AssertionError("expected PermissionError")


def test_artifact_rejects_unknown_key(tmp_path: Path) -> None:
    f = tmp_path / "runs" / "query" / "x.md"
    f.parent.mkdir(parents=True)
    f.write_text("x", encoding="utf-8")
    details = {"artifacts": {"report_path": str(f)}}
    try:
        resolve_artifact_path(run_details=details, artifact_key="random_key", roots=[str(tmp_path / "runs")])
    except KeyError:
        pass
    else:
        raise AssertionError("expected KeyError")
