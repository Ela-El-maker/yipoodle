from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import json
import time
from typing import Any

from src.apps.evidence_extract import extract_from_papers_dir_with_db
from src.apps.index_builder import build_index_incremental
from src.apps.paper_ingest import init_db


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class _SeenFile:
    path: str
    mtime_ns: int
    size: int


def _snapshot_pdf_dir(watch_dir: Path) -> dict[str, _SeenFile]:
    out: dict[str, _SeenFile] = {}
    for p in sorted(watch_dir.glob("*.pdf")):
        try:
            st = p.stat()
        except OSError:
            continue
        out[str(p.resolve())] = _SeenFile(path=str(p.resolve()), mtime_ns=int(st.st_mtime_ns), size=int(st.st_size))
    return out


def _detect_new_files(prev: dict[str, _SeenFile], curr: dict[str, _SeenFile]) -> list[str]:
    changed: list[str] = []
    for path, info in curr.items():
        old = prev.get(path)
        if old is None:
            changed.append(path)
            continue
        if old.mtime_ns != info.mtime_ns or old.size != info.size:
            changed.append(path)
    return sorted(changed)


def run_watch_ingest(
    *,
    watch_dir: str,
    extracted_dir: str,
    db_path: str,
    index_path: str,
    once: bool = False,
    poll_interval_sec: float = 2.0,
    max_events: int = 0,
    min_text_chars: int = 200,
    out_path: str | None = None,
) -> dict[str, Any]:
    wd = Path(watch_dir)
    wd.mkdir(parents=True, exist_ok=True)
    Path(extracted_dir).mkdir(parents=True, exist_ok=True)
    Path(index_path).parent.mkdir(parents=True, exist_ok=True)
    init_db(db_path)

    prev = _snapshot_pdf_dir(wd)
    processed: list[str] = []
    loops = 0
    start = time.time()

    def _process_changes(changed: list[str]) -> dict[str, Any]:
        if not changed:
            return {}
        out: dict[str, Any] = {}
        try:
            out["extract"] = extract_from_papers_dir_with_db(
                papers_dir=str(wd),
                out_dir=extracted_dir,
                db_path=db_path,
                min_text_chars=min_text_chars,
            )
        except Exception as exc:
            out["extract"] = {"error": str(exc), "failed": True}
            out["index"] = {"skipped": True, "reason": "extract_failed"}
            return out
        try:
            out["index"] = build_index_incremental(
                corpus_dir=extracted_dir,
                out_path=index_path,
                db_path=db_path,
            )
        except Exception as exc:
            out["index"] = {"error": str(exc), "failed": True}
        return out

    last_stage_stats: dict[str, Any] = {}
    while True:
        loops += 1
        curr = _snapshot_pdf_dir(wd)
        changed = _detect_new_files(prev, curr)
        if once and loops == 1 and not changed and curr:
            changed = sorted(curr.keys())
        prev = curr

        if changed:
            processed.extend(changed)
            last_stage_stats = _process_changes(changed)
            if max_events > 0 and len(processed) >= int(max_events):
                break

        if once:
            break
        time.sleep(max(0.2, float(poll_interval_sec)))

    payload = {
        "watch_dir": str(wd.resolve()),
        "extracted_dir": str(Path(extracted_dir).resolve()),
        "db_path": str(Path(db_path).resolve()),
        "index_path": str(Path(index_path).resolve()),
        "once": bool(once),
        "poll_interval_sec": float(poll_interval_sec),
        "max_events": int(max_events),
        "loops": loops,
        "processed_files": sorted(set(processed)),
        "processed_count": len(set(processed)),
        "elapsed_sec": round(time.time() - start, 3),
        "created_at": _utc_now(),
        "stages": last_stage_stats,
    }
    if out_path:
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload
