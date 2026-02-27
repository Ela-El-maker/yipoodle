from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import json
import sqlite3


RUN_STATUSES = {"queued", "running", "done", "failed", "cancelled"}


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


class UIRunStore:
    def __init__(self, db_path: str):
        self.db_path = db_path

    @contextmanager
    def _connect(self):
        p = Path(self.db_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(p), timeout=30)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys=ON")
            yield conn
        finally:
            conn.close()

    def init(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS ui_run (
                    run_id TEXT PRIMARY KEY,
                    mode TEXT NOT NULL,
                    question TEXT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    started_at TEXT NULL,
                    finished_at TEXT NULL,
                    error_message TEXT NULL,
                    output_path TEXT NULL,
                    details_json TEXT NOT NULL DEFAULT '{}',
                    cancel_requested INTEGER NOT NULL DEFAULT 0
                );

                CREATE TABLE IF NOT EXISTS ui_run_event (
                    seq INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    level TEXT NOT NULL,
                    message TEXT NOT NULL,
                    payload_json TEXT NOT NULL DEFAULT '{}',
                    FOREIGN KEY(run_id) REFERENCES ui_run(run_id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_ui_run_status_created ON ui_run(status, created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_ui_run_mode_created ON ui_run(mode, created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_ui_run_event_run_seq ON ui_run_event(run_id, seq);
                """
            )
            conn.commit()

    def create_run(
        self,
        *,
        run_id: str,
        mode: str,
        question: str | None,
        output_path: str | None,
        details: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        now = _utcnow()
        payload = json.dumps(details or {}, ensure_ascii=True)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO ui_run(
                    run_id, mode, question, status, created_at,
                    output_path, details_json, cancel_requested
                ) VALUES(?, ?, ?, 'queued', ?, ?, ?, 0)
                """,
                (run_id, mode, question, now, output_path, payload),
            )
            conn.commit()
        self.add_event(run_id=run_id, level="info", message="run_queued", payload={"mode": mode})
        return self.get_run(run_id)

    def get_run(self, run_id: str) -> dict[str, Any]:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM ui_run WHERE run_id = ?", (run_id,)).fetchone()
        if row is None:
            raise KeyError(f"run not found: {run_id}")
        return self._row_to_run(row)

    def list_runs(
        self,
        *,
        status: str | None = None,
        mode: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        where: list[str] = []
        params: list[Any] = []
        if status:
            where.append("status = ?")
            params.append(status)
        if mode:
            where.append("mode = ?")
            params.append(mode)
        sql = "SELECT * FROM ui_run"
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
        params.extend([int(limit), int(offset)])
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [self._row_to_run(r) for r in rows]

    def mark_running(self, run_id: str) -> None:
        now = _utcnow()
        with self._connect() as conn:
            conn.execute(
                "UPDATE ui_run SET status='running', started_at=?, error_message=NULL WHERE run_id=?",
                (now, run_id),
            )
            conn.commit()
        self.add_event(run_id=run_id, level="info", message="run_started", payload={})

    def mark_done(self, run_id: str, *, output_path: str | None, details: dict[str, Any]) -> None:
        now = _utcnow()
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE ui_run
                SET status='done', finished_at=?, error_message=NULL,
                    output_path=?, details_json=?
                WHERE run_id=?
                """,
                (now, output_path, json.dumps(details, ensure_ascii=True), run_id),
            )
            conn.commit()
        self.add_event(run_id=run_id, level="info", message="run_completed", payload={"output_path": output_path})

    def mark_failed(self, run_id: str, error_message: str, *, details: dict[str, Any] | None = None) -> None:
        now = _utcnow()
        with self._connect() as conn:
            if details is None:
                conn.execute(
                    "UPDATE ui_run SET status='failed', finished_at=?, error_message=? WHERE run_id=?",
                    (now, error_message, run_id),
                )
            else:
                conn.execute(
                    "UPDATE ui_run SET status='failed', finished_at=?, error_message=?, details_json=? WHERE run_id=?",
                    (now, error_message, json.dumps(details, ensure_ascii=True), run_id),
                )
            conn.commit()
        self.add_event(run_id=run_id, level="error", message="run_failed", payload={"error": error_message})

    def mark_cancelled(self, run_id: str, *, details: dict[str, Any] | None = None) -> None:
        now = _utcnow()
        with self._connect() as conn:
            if details is None:
                conn.execute(
                    "UPDATE ui_run SET status='cancelled', finished_at=? WHERE run_id=?",
                    (now, run_id),
                )
            else:
                conn.execute(
                    "UPDATE ui_run SET status='cancelled', finished_at=?, details_json=? WHERE run_id=?",
                    (now, json.dumps(details, ensure_ascii=True), run_id),
                )
            conn.commit()
        self.add_event(run_id=run_id, level="warn", message="run_cancelled", payload={})

    def request_cancel(self, run_id: str) -> bool:
        with self._connect() as conn:
            row = conn.execute("SELECT status FROM ui_run WHERE run_id=?", (run_id,)).fetchone()
            if row is None:
                raise KeyError(f"run not found: {run_id}")
            status = str(row["status"])
            conn.execute("UPDATE ui_run SET cancel_requested=1 WHERE run_id=?", (run_id,))
            conn.commit()
        self.add_event(run_id=run_id, level="warn", message="cancel_requested", payload={"status": status})
        return status in {"queued", "running"}

    def is_cancel_requested(self, run_id: str) -> bool:
        with self._connect() as conn:
            row = conn.execute("SELECT cancel_requested FROM ui_run WHERE run_id=?", (run_id,)).fetchone()
        return bool(row and int(row["cancel_requested"]) == 1)

    def add_event(self, *, run_id: str, level: str, message: str, payload: dict[str, Any]) -> int:
        now = _utcnow()
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO ui_run_event(run_id, created_at, level, message, payload_json) VALUES(?, ?, ?, ?, ?)",
                (run_id, now, level, message, json.dumps(payload, ensure_ascii=True)),
            )
            seq_row = conn.execute("SELECT last_insert_rowid() AS seq").fetchone()
            conn.commit()
        return int(seq_row["seq"])

    def list_events_since(self, run_id: str, after_seq: int = 0, limit: int = 200) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT seq, run_id, created_at, level, message, payload_json
                FROM ui_run_event
                WHERE run_id = ? AND seq > ?
                ORDER BY seq ASC
                LIMIT ?
                """,
                (run_id, int(after_seq), int(limit)),
            ).fetchall()
        out: list[dict[str, Any]] = []
        for r in rows:
            payload_raw = r["payload_json"]
            try:
                payload = json.loads(payload_raw) if payload_raw else {}
            except Exception:
                payload = {}
            out.append(
                {
                    "seq": int(r["seq"]),
                    "run_id": str(r["run_id"]),
                    "created_at": str(r["created_at"]),
                    "level": str(r["level"]),
                    "message": str(r["message"]),
                    "payload": payload if isinstance(payload, dict) else {},
                }
            )
        return out

    def recover_stale_running(self) -> int:
        now = _utcnow()
        with self._connect() as conn:
            rows = conn.execute("SELECT run_id FROM ui_run WHERE status='running'").fetchall()
            run_ids = [str(r["run_id"]) for r in rows]
            if run_ids:
                conn.execute(
                    "UPDATE ui_run SET status='failed', finished_at=?, error_message='stale_after_restart' WHERE status='running'",
                    (now,),
                )
                conn.commit()
        for run_id in run_ids:
            self.add_event(
                run_id=run_id,
                level="error",
                message="ui_recovery_stale_run",
                payload={"error": "stale_after_restart"},
            )
        return len(run_ids)

    def _row_to_run(self, row: sqlite3.Row) -> dict[str, Any]:
        try:
            details = json.loads(row["details_json"] or "{}")
        except Exception:
            details = {}
        if not isinstance(details, dict):
            details = {}
        status = str(row["status"])
        if status not in RUN_STATUSES:
            status = "failed"
        return {
            "run_id": str(row["run_id"]),
            "mode": str(row["mode"]),
            "question": row["question"],
            "status": status,
            "created_at": str(row["created_at"]),
            "started_at": row["started_at"],
            "finished_at": row["finished_at"],
            "error_message": row["error_message"],
            "output_path": row["output_path"],
            "details": details,
            "cancel_requested": bool(int(row["cancel_requested"])),
        }
