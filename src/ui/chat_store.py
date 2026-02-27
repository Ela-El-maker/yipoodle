from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
import json
import sqlite3


CHAT_MESSAGE_STATUSES = {"queued", "running", "done", "failed", "cancelled"}
CHAT_MESSAGE_ROLES = {"user", "assistant", "system"}


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


class UIChatStore:
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
                CREATE TABLE IF NOT EXISTS ui_chat_session (
                    id TEXT PRIMARY KEY,
                    title TEXT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    archived INTEGER NOT NULL DEFAULT 0
                );

                CREATE TABLE IF NOT EXISTS ui_chat_message (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    mode_requested TEXT NULL,
                    mode_used TEXT NULL,
                    status TEXT NOT NULL,
                    content_markdown TEXT NOT NULL DEFAULT '',
                    error_message TEXT NULL,
                    created_at TEXT NOT NULL,
                    started_at TEXT NULL,
                    finished_at TEXT NULL,
                    run_id TEXT NULL,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    cancel_requested INTEGER NOT NULL DEFAULT 0,
                    FOREIGN KEY(session_id) REFERENCES ui_chat_session(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS ui_chat_message_blob (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    message_id TEXT NOT NULL,
                    blob_type TEXT NOT NULL,
                    blob_text TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(message_id) REFERENCES ui_chat_message(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS ui_chat_event (
                    seq INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    message_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    level TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    payload_json TEXT NOT NULL DEFAULT '{}',
                    FOREIGN KEY(session_id) REFERENCES ui_chat_session(id) ON DELETE CASCADE,
                    FOREIGN KEY(message_id) REFERENCES ui_chat_message(id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_ui_chat_session_updated ON ui_chat_session(updated_at DESC);
                CREATE INDEX IF NOT EXISTS idx_ui_chat_message_session_created ON ui_chat_message(session_id, created_at ASC);
                CREATE INDEX IF NOT EXISTS idx_ui_chat_message_status_created ON ui_chat_message(status, created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_ui_chat_event_session_seq ON ui_chat_event(session_id, seq);
                CREATE INDEX IF NOT EXISTS idx_ui_chat_event_message_seq ON ui_chat_event(message_id, seq);
                """
            )
            conn.commit()

    def create_session(self, *, session_id: str, title: str | None = None) -> dict[str, Any]:
        now = _utcnow()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO ui_chat_session(id, title, created_at, updated_at, archived)
                VALUES(?, ?, ?, ?, 0)
                """,
                (session_id, title, now, now),
            )
            conn.commit()
        return self.get_session(session_id)

    def list_sessions(self, *, limit: int = 100, offset: int = 0, include_archived: bool = False) -> list[dict[str, Any]]:
        where = "" if include_archived else "WHERE archived = 0"
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT id, title, created_at, updated_at, archived
                FROM ui_chat_session
                {where}
                ORDER BY updated_at DESC
                LIMIT ? OFFSET ?
                """,
                (int(limit), int(offset)),
            ).fetchall()
        return [self._row_to_session(r) for r in rows]

    def get_session(self, session_id: str) -> dict[str, Any]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT id, title, created_at, updated_at, archived FROM ui_chat_session WHERE id = ?",
                (session_id,),
            ).fetchone()
        if row is None:
            raise KeyError(f"session not found: {session_id}")
        return self._row_to_session(row)

    def update_session_title(self, *, session_id: str, title: str) -> None:
        now = _utcnow()
        with self._connect() as conn:
            conn.execute(
                "UPDATE ui_chat_session SET title = ?, updated_at = ? WHERE id = ?",
                (title, now, session_id),
            )
            conn.commit()

    def touch_session(self, session_id: str) -> None:
        now = _utcnow()
        with self._connect() as conn:
            conn.execute(
                "UPDATE ui_chat_session SET updated_at = ? WHERE id = ?",
                (now, session_id),
            )
            conn.commit()

    def create_message(
        self,
        *,
        message_id: str,
        session_id: str,
        role: str,
        mode_requested: str | None,
        status: str,
        content_markdown: str,
        metadata: dict[str, Any] | None = None,
        mode_used: str | None = None,
        run_id: str | None = None,
        error_message: str | None = None,
    ) -> dict[str, Any]:
        if role not in CHAT_MESSAGE_ROLES:
            raise ValueError(f"invalid role: {role}")
        if status not in CHAT_MESSAGE_STATUSES:
            raise ValueError(f"invalid status: {status}")
        now = _utcnow()
        metadata_json = json.dumps(metadata or {}, ensure_ascii=True)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO ui_chat_message(
                    id, session_id, role, mode_requested, mode_used, status,
                    content_markdown, error_message, created_at, started_at, finished_at,
                    run_id, metadata_json, cancel_requested
                ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL, ?, ?, 0)
                """,
                (
                    message_id,
                    session_id,
                    role,
                    mode_requested,
                    mode_used,
                    status,
                    content_markdown,
                    error_message,
                    now,
                    run_id,
                    metadata_json,
                ),
            )
            conn.execute(
                "UPDATE ui_chat_session SET updated_at = ? WHERE id = ?",
                (now, session_id),
            )
            conn.commit()
        return self.get_message(message_id)

    def get_message(self, message_id: str) -> dict[str, Any]:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM ui_chat_message WHERE id = ?", (message_id,)).fetchone()
        if row is None:
            raise KeyError(f"message not found: {message_id}")
        return self._row_to_message(row)

    def list_messages(self, session_id: str, *, limit: int = 500, offset: int = 0) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM ui_chat_message
                WHERE session_id = ?
                ORDER BY created_at ASC
                LIMIT ? OFFSET ?
                """,
                (session_id, int(limit), int(offset)),
            ).fetchall()
        return [self._row_to_message(r) for r in rows]

    def mark_running(self, message_id: str) -> None:
        now = _utcnow()
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE ui_chat_message
                SET status='running', started_at=?, error_message=NULL
                WHERE id=?
                """,
                (now, message_id),
            )
            conn.commit()

    def mark_done(
        self,
        message_id: str,
        *,
        mode_used: str,
        content_markdown: str,
        metadata: dict[str, Any],
    ) -> None:
        now = _utcnow()
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE ui_chat_message
                SET status='done', finished_at=?, mode_used=?, content_markdown=?, metadata_json=?
                WHERE id=?
                """,
                (now, mode_used, content_markdown, json.dumps(metadata, ensure_ascii=True), message_id),
            )
            session_row = conn.execute("SELECT session_id FROM ui_chat_message WHERE id=?", (message_id,)).fetchone()
            if session_row:
                conn.execute(
                    "UPDATE ui_chat_session SET updated_at = ? WHERE id = ?",
                    (now, str(session_row["session_id"])),
                )
            conn.commit()

    def mark_failed(self, message_id: str, error_message: str, *, metadata: dict[str, Any] | None = None) -> None:
        now = _utcnow()
        with self._connect() as conn:
            if metadata is None:
                conn.execute(
                    "UPDATE ui_chat_message SET status='failed', finished_at=?, error_message=? WHERE id=?",
                    (now, error_message, message_id),
                )
            else:
                conn.execute(
                    """
                    UPDATE ui_chat_message
                    SET status='failed', finished_at=?, error_message=?, metadata_json=?
                    WHERE id=?
                    """,
                    (now, error_message, json.dumps(metadata, ensure_ascii=True), message_id),
                )
            session_row = conn.execute("SELECT session_id FROM ui_chat_message WHERE id=?", (message_id,)).fetchone()
            if session_row:
                conn.execute(
                    "UPDATE ui_chat_session SET updated_at = ? WHERE id = ?",
                    (now, str(session_row["session_id"])),
                )
            conn.commit()

    def mark_cancelled(self, message_id: str, *, metadata: dict[str, Any] | None = None) -> None:
        now = _utcnow()
        with self._connect() as conn:
            if metadata is None:
                conn.execute(
                    "UPDATE ui_chat_message SET status='cancelled', finished_at=? WHERE id=?",
                    (now, message_id),
                )
            else:
                conn.execute(
                    """
                    UPDATE ui_chat_message
                    SET status='cancelled', finished_at=?, metadata_json=?
                    WHERE id=?
                    """,
                    (now, json.dumps(metadata, ensure_ascii=True), message_id),
                )
            session_row = conn.execute("SELECT session_id FROM ui_chat_message WHERE id=?", (message_id,)).fetchone()
            if session_row:
                conn.execute(
                    "UPDATE ui_chat_session SET updated_at = ? WHERE id = ?",
                    (now, str(session_row["session_id"])),
                )
            conn.commit()

    def request_cancel(self, message_id: str) -> bool:
        with self._connect() as conn:
            row = conn.execute("SELECT status FROM ui_chat_message WHERE id=?", (message_id,)).fetchone()
            if row is None:
                raise KeyError(f"message not found: {message_id}")
            status = str(row["status"])
            conn.execute("UPDATE ui_chat_message SET cancel_requested=1 WHERE id=?", (message_id,))
            conn.commit()
        return status in {"queued", "running"}

    def is_cancel_requested(self, message_id: str) -> bool:
        with self._connect() as conn:
            row = conn.execute("SELECT cancel_requested FROM ui_chat_message WHERE id=?", (message_id,)).fetchone()
        return bool(row and int(row["cancel_requested"]) == 1)

    def add_blob(self, *, message_id: str, blob_type: str, blob_text: str) -> int:
        now = _utcnow()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO ui_chat_message_blob(message_id, blob_type, blob_text, created_at)
                VALUES(?, ?, ?, ?)
                """,
                (message_id, blob_type, blob_text, now),
            )
            row = conn.execute("SELECT last_insert_rowid() AS id").fetchone()
            conn.commit()
        return int(row["id"])

    def list_blobs(self, message_id: str) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, message_id, blob_type, blob_text, created_at
                FROM ui_chat_message_blob
                WHERE message_id = ?
                ORDER BY id ASC
                """,
                (message_id,),
            ).fetchall()
        out: list[dict[str, Any]] = []
        for r in rows:
            out.append(
                {
                    "id": int(r["id"]),
                    "message_id": str(r["message_id"]),
                    "blob_type": str(r["blob_type"]),
                    "blob_text": str(r["blob_text"]),
                    "created_at": str(r["created_at"]),
                }
            )
        return out

    def add_event(
        self,
        *,
        session_id: str,
        message_id: str,
        level: str,
        event_type: str,
        payload: dict[str, Any],
    ) -> int:
        now = _utcnow()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO ui_chat_event(
                    session_id, message_id, created_at, level, event_type, payload_json
                ) VALUES(?, ?, ?, ?, ?, ?)
                """,
                (session_id, message_id, now, level, event_type, json.dumps(payload, ensure_ascii=True)),
            )
            row = conn.execute("SELECT last_insert_rowid() AS seq").fetchone()
            conn.execute(
                "UPDATE ui_chat_session SET updated_at = ? WHERE id = ?",
                (now, session_id),
            )
            conn.commit()
        return int(row["seq"])

    def list_events_since(
        self,
        session_id: str,
        *,
        after_seq: int = 0,
        limit: int = 200,
        message_id: str | None = None,
    ) -> list[dict[str, Any]]:
        where = "WHERE session_id = ? AND seq > ?"
        params: list[Any] = [session_id, int(after_seq)]
        if message_id:
            where += " AND message_id = ?"
            params.append(message_id)
        params.append(int(limit))
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT seq, session_id, message_id, created_at, level, event_type, payload_json
                FROM ui_chat_event
                {where}
                ORDER BY seq ASC
                LIMIT ?
                """,
                params,
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
                    "session_id": str(r["session_id"]),
                    "message_id": str(r["message_id"]),
                    "created_at": str(r["created_at"]),
                    "level": str(r["level"]),
                    "event_type": str(r["event_type"]),
                    "payload": payload if isinstance(payload, dict) else {},
                }
            )
        return out

    def recover_stale_running(self) -> int:
        now = _utcnow()
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, session_id FROM ui_chat_message WHERE status='running'"
            ).fetchall()
            stale = [(str(r["id"]), str(r["session_id"])) for r in rows]
            if stale:
                conn.execute(
                    """
                    UPDATE ui_chat_message
                    SET status='failed', finished_at=?, error_message='stale_after_restart'
                    WHERE status='running'
                    """,
                    (now,),
                )
                conn.commit()
        for message_id, session_id in stale:
            self.add_event(
                session_id=session_id,
                message_id=message_id,
                level="error",
                event_type="ui_recovery_stale_message",
                payload={"error": "stale_after_restart"},
            )
        return len(stale)

    def prune_events_older_than(self, *, days: int) -> int:
        cutoff = (datetime.now(timezone.utc) - timedelta(days=max(1, int(days)))).isoformat()
        with self._connect() as conn:
            row = conn.execute("SELECT COUNT(*) AS n FROM ui_chat_event WHERE created_at < ?", (cutoff,)).fetchone()
            count = int(row["n"]) if row else 0
            conn.execute("DELETE FROM ui_chat_event WHERE created_at < ?", (cutoff,))
            conn.commit()
        return count

    @staticmethod
    def _row_to_session(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "id": str(row["id"]),
            "title": row["title"],
            "created_at": str(row["created_at"]),
            "updated_at": str(row["updated_at"]),
            "archived": bool(int(row["archived"])),
        }

    @staticmethod
    def _row_to_message(row: sqlite3.Row) -> dict[str, Any]:
        metadata_raw = row["metadata_json"]
        try:
            metadata = json.loads(metadata_raw) if metadata_raw else {}
        except Exception:
            metadata = {}
        return {
            "id": str(row["id"]),
            "session_id": str(row["session_id"]),
            "role": str(row["role"]),
            "mode_requested": row["mode_requested"],
            "mode_used": row["mode_used"],
            "status": str(row["status"]),
            "content_markdown": str(row["content_markdown"] or ""),
            "error_message": row["error_message"],
            "created_at": str(row["created_at"]),
            "started_at": row["started_at"],
            "finished_at": row["finished_at"],
            "run_id": row["run_id"],
            "metadata": metadata if isinstance(metadata, dict) else {},
            "cancel_requested": bool(int(row["cancel_requested"])),
        }
