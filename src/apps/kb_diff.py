from __future__ import annotations

from typing import Any

from src.apps.kb_store import changes_since, connect_kb, init_kb


def run_kb_diff(*, kb_db: str, topic: str, since_run: str | None) -> dict[str, Any]:
    init_kb(kb_db)
    with connect_kb(kb_db) as conn:
        rows = changes_since(conn, topic=topic, since_run=since_run)
    counts = {"added": 0, "updated": 0, "disputed": 0, "superseded": 0}
    for row in rows:
        diff = row.get("diff", {}) if isinstance(row, dict) else {}
        c = diff.get("counts", {}) if isinstance(diff, dict) else {}
        for k in counts:
            counts[k] += int(c.get(k, 0) or 0)
    return {
        "kb_db": kb_db,
        "topic": topic,
        "since_run": since_run,
        "entries": rows,
        "aggregate_counts": counts,
    }
