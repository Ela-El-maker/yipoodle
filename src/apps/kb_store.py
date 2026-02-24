from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
import json
import sqlite3
from typing import Any
import re

from src.apps.kb_confidence import apply_confidence_decay, contradiction_score, parse_iso_utc


SCHEMA_SQL = """
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS kb_topic (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  name TEXT NOT NULL UNIQUE,
  parent_id INTEGER NULL,
  tags_json TEXT NOT NULL DEFAULT '[]',
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  FOREIGN KEY(parent_id) REFERENCES kb_topic(id)
);

CREATE TABLE IF NOT EXISTS kb_claim (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  topic_id INTEGER NOT NULL,
  claim_text TEXT NOT NULL,
  canonical_hash TEXT NOT NULL,
  status TEXT NOT NULL DEFAULT 'active',
  confidence REAL NOT NULL DEFAULT 0.0,
  first_seen_at TEXT NOT NULL,
  last_confirmed_at TEXT NOT NULL,
  last_seen_run_id TEXT NOT NULL,
  UNIQUE(topic_id, canonical_hash),
  FOREIGN KEY(topic_id) REFERENCES kb_topic(id)
);

CREATE INDEX IF NOT EXISTS idx_kb_claim_topic_status ON kb_claim(topic_id, status);
CREATE INDEX IF NOT EXISTS idx_kb_claim_last_confirmed ON kb_claim(last_confirmed_at);

CREATE TABLE IF NOT EXISTS kb_claim_version (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  claim_id INTEGER NOT NULL,
  claim_text TEXT NOT NULL,
  confidence REAL NOT NULL,
  created_at TEXT NOT NULL,
  run_id TEXT NOT NULL,
  reason TEXT NOT NULL,
  FOREIGN KEY(claim_id) REFERENCES kb_claim(id)
);

CREATE TABLE IF NOT EXISTS kb_evidence (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  claim_id INTEGER NOT NULL,
  snippet_id TEXT NOT NULL,
  source_type TEXT NOT NULL,
  support_score REAL NOT NULL,
  retrieved_at TEXT NOT NULL,
  run_id TEXT NOT NULL,
  UNIQUE(claim_id, snippet_id, run_id),
  FOREIGN KEY(claim_id) REFERENCES kb_claim(id)
);

CREATE TABLE IF NOT EXISTS kb_contradiction (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  claim_id_a INTEGER NOT NULL,
  claim_id_b INTEGER NOT NULL,
  score REAL NOT NULL,
  detected_at TEXT NOT NULL,
  run_id TEXT NOT NULL,
  reason TEXT NOT NULL,
  UNIQUE(claim_id_a, claim_id_b, run_id),
  FOREIGN KEY(claim_id_a) REFERENCES kb_claim(id),
  FOREIGN KEY(claim_id_b) REFERENCES kb_claim(id)
);

CREATE TABLE IF NOT EXISTS kb_note (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  topic_id INTEGER NOT NULL,
  note_path TEXT NOT NULL,
  run_id TEXT NOT NULL,
  created_at TEXT NOT NULL,
  evidence_path TEXT,
  metrics_path TEXT,
  FOREIGN KEY(topic_id) REFERENCES kb_topic(id)
);

CREATE TABLE IF NOT EXISTS kb_change (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  topic_id INTEGER NOT NULL,
  run_id TEXT NOT NULL,
  created_at TEXT NOT NULL,
  diff_json TEXT NOT NULL,
  FOREIGN KEY(topic_id) REFERENCES kb_topic(id)
);

CREATE INDEX IF NOT EXISTS idx_kb_change_topic_created ON kb_change(topic_id, created_at);
CREATE INDEX IF NOT EXISTS idx_kb_change_topic_run ON kb_change(topic_id, run_id);

CREATE VIRTUAL TABLE IF NOT EXISTS kb_claim_fts USING fts5(claim_text);
"""


@contextmanager
def connect_kb(db_path: str):
    p = Path(db_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(p))
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA foreign_keys=ON")
        yield conn
    finally:
        conn.close()


def init_kb(db_path: str) -> None:
    with connect_kb(db_path) as conn:
        conn.executescript(SCHEMA_SQL)
        conn.commit()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_topic(conn: sqlite3.Connection, name: str, *, tags: list[str] | None = None) -> int:
    now = _utc_now()
    row = conn.execute("SELECT id FROM kb_topic WHERE name = ?", (name,)).fetchone()
    if row:
        tid = int(row[0])
        conn.execute("UPDATE kb_topic SET updated_at = ? WHERE id = ?", (now, tid))
        return tid
    conn.execute(
        "INSERT INTO kb_topic(name, parent_id, tags_json, created_at, updated_at) VALUES(?, NULL, ?, ?, ?)",
        (name, json.dumps(tags or []), now, now),
    )
    return int(conn.execute("SELECT last_insert_rowid()").fetchone()[0])


def list_topics(conn: sqlite3.Connection) -> list[str]:
    rows = conn.execute("SELECT name FROM kb_topic ORDER BY name").fetchall()
    return [str(r[0]) for r in rows]


def get_topic_id(conn: sqlite3.Connection, name: str) -> int | None:
    row = conn.execute("SELECT id FROM kb_topic WHERE name = ?", (name,)).fetchone()
    return int(row[0]) if row else None


def _fts_upsert(conn: sqlite3.Connection, claim_id: int, claim_text: str) -> None:
    conn.execute("DELETE FROM kb_claim_fts WHERE rowid = ?", (claim_id,))
    conn.execute("INSERT INTO kb_claim_fts(rowid, claim_text) VALUES(?, ?)", (claim_id, claim_text))


def upsert_claim(
    conn: sqlite3.Connection,
    *,
    topic_id: int,
    claim_text: str,
    canonical_hash: str,
    confidence: float,
    run_id: str,
    now_iso: str,
) -> tuple[int, str, float, str]:
    row = conn.execute(
        "SELECT id, confidence, status FROM kb_claim WHERE topic_id = ? AND canonical_hash = ?",
        (topic_id, canonical_hash),
    ).fetchone()
    if row is None:
        conn.execute(
            """
            INSERT INTO kb_claim(topic_id, claim_text, canonical_hash, status, confidence, first_seen_at, last_confirmed_at, last_seen_run_id)
            VALUES(?, ?, ?, 'active', ?, ?, ?, ?)
            """,
            (topic_id, claim_text, canonical_hash, float(confidence), now_iso, now_iso, run_id),
        )
        claim_id = int(conn.execute("SELECT last_insert_rowid()").fetchone()[0])
        _fts_upsert(conn, claim_id, claim_text)
        return claim_id, "added", 0.0, "active"

    claim_id = int(row["id"])
    prev_conf = float(row["confidence"])
    prev_status = str(row["status"])
    conn.execute(
        """
        UPDATE kb_claim
        SET claim_text = ?, confidence = ?, last_confirmed_at = ?, last_seen_run_id = ?,
            status = CASE WHEN status = 'superseded' THEN 'superseded' ELSE 'active' END
        WHERE id = ?
        """,
        (claim_text, float(confidence), now_iso, run_id, claim_id),
    )
    _fts_upsert(conn, claim_id, claim_text)
    return claim_id, "updated", prev_conf, prev_status


def insert_claim_version(
    conn: sqlite3.Connection,
    *,
    claim_id: int,
    claim_text: str,
    confidence: float,
    run_id: str,
    reason: str,
    now_iso: str,
) -> None:
    conn.execute(
        "INSERT INTO kb_claim_version(claim_id, claim_text, confidence, created_at, run_id, reason) VALUES(?, ?, ?, ?, ?, ?)",
        (claim_id, claim_text, float(confidence), now_iso, run_id, reason),
    )


def insert_evidence(
    conn: sqlite3.Connection,
    *,
    claim_id: int,
    snippet_id: str,
    source_type: str,
    support_score: float,
    retrieved_at: str,
    run_id: str,
) -> None:
    conn.execute(
        """
        INSERT OR IGNORE INTO kb_evidence(claim_id, snippet_id, source_type, support_score, retrieved_at, run_id)
        VALUES(?, ?, ?, ?, ?, ?)
        """,
        (claim_id, snippet_id, source_type, float(support_score), retrieved_at, run_id),
    )


def mark_disputed(conn: sqlite3.Connection, claim_id: int) -> None:
    conn.execute("UPDATE kb_claim SET status = 'disputed' WHERE id = ?", (claim_id,))


def find_potential_contradictions(
    conn: sqlite3.Connection,
    *,
    topic_id: int,
    claim_id: int,
    claim_text: str,
    threshold: float = 0.65,
) -> list[tuple[int, float, str]]:
    rows = conn.execute(
        "SELECT id, claim_text FROM kb_claim WHERE topic_id = ? AND id != ? AND status IN ('active', 'disputed')",
        (topic_id, claim_id),
    ).fetchall()
    out: list[tuple[int, float, str]] = []
    for r in rows:
        oid = int(r["id"])
        score = contradiction_score(claim_text, str(r["claim_text"]))
        if score >= float(threshold):
            out.append((oid, score, "negation_overlap_conflict"))
    return out


def insert_contradiction(
    conn: sqlite3.Connection,
    *,
    claim_id_a: int,
    claim_id_b: int,
    score: float,
    run_id: str,
    reason: str,
    now_iso: str,
) -> None:
    a, b = sorted((int(claim_id_a), int(claim_id_b)))
    conn.execute(
        """
        INSERT OR IGNORE INTO kb_contradiction(claim_id_a, claim_id_b, score, detected_at, run_id, reason)
        VALUES(?, ?, ?, ?, ?, ?)
        """,
        (a, b, float(score), now_iso, run_id, reason),
    )


def add_note(
    conn: sqlite3.Connection,
    *,
    topic_id: int,
    note_path: str,
    run_id: str,
    evidence_path: str | None,
    metrics_path: str | None,
    now_iso: str,
) -> None:
    conn.execute(
        "INSERT INTO kb_note(topic_id, note_path, run_id, created_at, evidence_path, metrics_path) VALUES(?, ?, ?, ?, ?, ?)",
        (topic_id, note_path, run_id, now_iso, evidence_path, metrics_path),
    )


def record_change(conn: sqlite3.Connection, *, topic_id: int, run_id: str, diff: dict[str, Any], now_iso: str) -> None:
    conn.execute(
        "INSERT INTO kb_change(topic_id, run_id, created_at, diff_json) VALUES(?, ?, ?, ?)",
        (topic_id, run_id, now_iso, json.dumps(diff, sort_keys=True)),
    )


def decay_topic_confidence(
    conn: sqlite3.Connection,
    *,
    topic_id: int,
    now_iso: str,
    decay_per_day: float = 0.98,
    stale_threshold: float = 0.35,
) -> int:
    now_dt = parse_iso_utc(now_iso)
    if now_dt is None:
        return 0
    rows = conn.execute(
        "SELECT id, confidence, last_confirmed_at, status FROM kb_claim WHERE topic_id = ? AND status != 'superseded'",
        (topic_id,),
    ).fetchall()
    updated = 0
    for r in rows:
        cid = int(r["id"])
        last_dt = parse_iso_utc(str(r["last_confirmed_at"]))
        if last_dt is None:
            continue
        delta_days = (now_dt - last_dt).total_seconds() / 86400.0
        if delta_days <= 0.0:
            continue
        cur_conf = float(r["confidence"])
        decayed = apply_confidence_decay(cur_conf, days_since_confirmed=delta_days, decay_per_day=decay_per_day)
        new_status = "stale" if decayed < float(stale_threshold) else str(r["status"])
        if decayed != cur_conf or new_status != str(r["status"]):
            conn.execute("UPDATE kb_claim SET confidence = ?, status = ? WHERE id = ?", (decayed, new_status, cid))
            updated += 1
    return updated


def query_claims(
    conn: sqlite3.Connection,
    *,
    query: str,
    topic: str | None,
    top_k: int,
) -> list[dict[str, Any]]:
    params: list[Any] = []
    where = ["1=1"]
    if topic:
        where.append("t.name = ?")
        params.append(topic)

    base_sql = (
        "SELECT c.id, t.name AS topic, c.claim_text, c.status, c.confidence, c.last_confirmed_at "
        "FROM kb_claim c JOIN kb_topic t ON t.id = c.topic_id "
    )

    rows: list[sqlite3.Row]
    if query.strip():
        toks = [t for t in re.findall(r"[A-Za-z0-9]+", query.lower()) if t]
        match_q = " OR ".join(toks) if toks else query.strip()
        where_sql = " AND ".join(where)
        sql = (
            "SELECT c.id, t.name AS topic, c.claim_text, c.status, c.confidence, c.last_confirmed_at, "
            "bm25(kb_claim_fts) AS rank "
            "FROM kb_claim_fts f "
            "JOIN kb_claim c ON c.id = f.rowid "
            "JOIN kb_topic t ON t.id = c.topic_id "
            f"WHERE {where_sql} AND f.kb_claim_fts MATCH ? "
            "ORDER BY rank ASC, c.confidence DESC LIMIT ?"
        )
        params_q = list(params) + [match_q, int(top_k)]
        rows = conn.execute(sql, params_q).fetchall()
    else:
        where_sql = " AND ".join(where)
        sql = base_sql + f"WHERE {where_sql} ORDER BY c.confidence DESC, c.last_confirmed_at DESC LIMIT ?"
        rows = conn.execute(sql, list(params) + [int(top_k)]).fetchall()

    out: list[dict[str, Any]] = []
    for r in rows:
        claim_id = int(r["id"])
        ev = conn.execute(
            "SELECT snippet_id, source_type, support_score FROM kb_evidence WHERE claim_id = ? ORDER BY support_score DESC LIMIT 10",
            (claim_id,),
        ).fetchall()
        out.append(
            {
                "claim_id": claim_id,
                "topic": str(r["topic"]),
                "claim_text": str(r["claim_text"]),
                "status": str(r["status"]),
                "confidence": float(r["confidence"]),
                "last_confirmed_at": str(r["last_confirmed_at"]),
                "evidence": [
                    {
                        "snippet_id": str(e["snippet_id"]),
                        "source_type": str(e["source_type"]),
                        "support_score": float(e["support_score"]),
                    }
                    for e in ev
                ],
            }
        )
    return out


def changes_since(conn: sqlite3.Connection, *, topic: str, since_run: str | None) -> list[dict[str, Any]]:
    topic_id = get_topic_id(conn, topic)
    if topic_id is None:
        return []

    if not since_run:
        rows = conn.execute(
            "SELECT run_id, created_at, diff_json FROM kb_change WHERE topic_id = ? ORDER BY created_at DESC LIMIT 20",
            (topic_id,),
        ).fetchall()
    else:
        run_row = conn.execute(
            "SELECT created_at FROM kb_change WHERE topic_id = ? AND run_id = ? ORDER BY created_at DESC LIMIT 1",
            (topic_id, since_run),
        ).fetchone()
        cutoff = None
        if run_row:
            cutoff = str(run_row["created_at"])
        else:
            dt = parse_iso_utc(since_run)
            cutoff = dt.isoformat() if dt is not None else None
        if cutoff is None:
            rows = []
        else:
            rows = conn.execute(
                "SELECT run_id, created_at, diff_json FROM kb_change WHERE topic_id = ? AND created_at > ? ORDER BY created_at ASC",
                (topic_id, cutoff),
            ).fetchall()

    out: list[dict[str, Any]] = []
    for r in rows:
        out.append(
            {
                "run_id": str(r["run_id"]),
                "created_at": str(r["created_at"]),
                "diff": json.loads(str(r["diff_json"])),
            }
        )
    return out
