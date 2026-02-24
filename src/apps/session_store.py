"""Research Session Tracking (Feature #4).

Groups multiple research queries into named sessions, persists session
metadata in SQLite, and provides cross-query session summaries.

Sessions allow researchers to:
  • Group related research queries under a named session.
  • Track all questions asked, report paths, key findings, and timings.
  • Generate cross-query summaries (accumulated gaps, overlapping citations,
    topic convergence).
  • Lifecycle management: create → record queries → summarize → close.

Storage follows the same SQLite patterns used by ``kb_store``:
context-manager connections, ``CREATE TABLE IF NOT EXISTS`` idempotent init,
``sqlite3.Row`` row factory, and UTC ISO-8601 timestamps throughout.
"""

from __future__ import annotations

import json
import sqlite3
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

__all__ = [
    "SessionInfo",
    "SessionQuery",
    "SessionDetail",
    "SessionSummary",
    "init_session_db",
    "create_session",
    "close_session",
    "reopen_session",
    "delete_session",
    "record_query",
    "list_sessions",
    "get_session_detail",
    "session_summary",
    "render_session_markdown",
    "DEFAULT_SESSION_DB",
]

DEFAULT_SESSION_DB = "data/sessions/sessions.db"

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class SessionInfo:
    """Lightweight session metadata returned by :func:`list_sessions`."""

    session_id: int
    name: str
    description: str
    status: str  # "open" | "closed"
    created_at: str
    closed_at: str | None
    query_count: int


@dataclass
class SessionQuery:
    """A single query recorded within a session."""

    query_id: int
    session_id: int
    question: str
    report_path: str
    created_at: str
    elapsed_ms: float | None = None
    key_claims: list[str] = field(default_factory=list)
    gaps: list[str] = field(default_factory=list)
    citations: list[str] = field(default_factory=list)
    synthesis_preview: str = ""
    evidence_count: int = 0
    aggregation_enabled: bool = False


@dataclass
class SessionDetail:
    """Full session with all associated queries."""

    info: SessionInfo
    queries: list[SessionQuery]


@dataclass
class SessionSummary:
    """Cross-query summary computed from all queries in a session."""

    session_id: int
    session_name: str
    query_count: int
    total_evidence_items: int
    unique_papers: list[str]
    total_citations: int
    unique_citations: list[str]
    all_gaps: list[str]
    recurring_gaps: list[str]  # gaps appearing in ≥2 queries
    all_key_claims: list[str]
    question_timeline: list[dict[str, str]]  # [{"question": ..., "created_at": ...}]
    avg_elapsed_ms: float | None
    aggregation_used_count: int
    coverage: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# SQLite schema & connection
# ---------------------------------------------------------------------------

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS session (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT    NOT NULL UNIQUE,
    description TEXT    NOT NULL DEFAULT '',
    status      TEXT    NOT NULL DEFAULT 'open',
    created_at  TEXT    NOT NULL,
    closed_at   TEXT    NULL
);

CREATE INDEX IF NOT EXISTS idx_session_status ON session(status);

CREATE TABLE IF NOT EXISTS session_query (
    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id           INTEGER NOT NULL,
    question             TEXT    NOT NULL,
    report_path          TEXT    NOT NULL,
    created_at           TEXT    NOT NULL,
    elapsed_ms           REAL    NULL,
    key_claims_json      TEXT    NOT NULL DEFAULT '[]',
    gaps_json            TEXT    NOT NULL DEFAULT '[]',
    citations_json       TEXT    NOT NULL DEFAULT '[]',
    synthesis_preview    TEXT    NOT NULL DEFAULT '',
    evidence_count       INTEGER NOT NULL DEFAULT 0,
    aggregation_enabled  INTEGER NOT NULL DEFAULT 0,
    FOREIGN KEY(session_id) REFERENCES session(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_session_query_session ON session_query(session_id);
"""


@contextmanager
def _connect(db_path: str):
    """Context-managed SQLite connection with Row factory and FK enforcement."""
    p = Path(db_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(p))
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA foreign_keys=ON")
        yield conn
    finally:
        conn.close()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Public API — database lifecycle
# ---------------------------------------------------------------------------


def init_session_db(db_path: str = DEFAULT_SESSION_DB) -> None:
    """Create the session tables if they don't exist."""
    with _connect(db_path) as conn:
        conn.executescript(_SCHEMA_SQL)
        conn.commit()


# ---------------------------------------------------------------------------
# Public API — session CRUD
# ---------------------------------------------------------------------------


def create_session(
    db_path: str,
    name: str,
    description: str = "",
) -> int:
    """Create a new research session and return its ID.

    Raises ``ValueError`` if *name* already exists.
    """
    init_session_db(db_path)
    now = _utc_now()
    with _connect(db_path) as conn:
        try:
            conn.execute(
                "INSERT INTO session(name, description, status, created_at) VALUES(?, ?, 'open', ?)",
                (name, description, now),
            )
            conn.commit()
            row = conn.execute("SELECT last_insert_rowid()").fetchone()
            return int(row[0])
        except sqlite3.IntegrityError:
            raise ValueError(f"Session '{name}' already exists") from None


def close_session(db_path: str, session_id: int) -> None:
    """Mark a session as closed.

    Raises ``ValueError`` if the session doesn't exist or is already closed.
    """
    with _connect(db_path) as conn:
        row = conn.execute("SELECT status FROM session WHERE id = ?", (session_id,)).fetchone()
        if row is None:
            raise ValueError(f"Session {session_id} not found")
        if row["status"] == "closed":
            raise ValueError(f"Session {session_id} is already closed")
        conn.execute(
            "UPDATE session SET status = 'closed', closed_at = ? WHERE id = ?",
            (_utc_now(), session_id),
        )
        conn.commit()


def reopen_session(db_path: str, session_id: int) -> None:
    """Reopen a previously closed session.

    Raises ``ValueError`` if the session doesn't exist or is already open.
    """
    with _connect(db_path) as conn:
        row = conn.execute("SELECT status FROM session WHERE id = ?", (session_id,)).fetchone()
        if row is None:
            raise ValueError(f"Session {session_id} not found")
        if row["status"] == "open":
            raise ValueError(f"Session {session_id} is already open")
        conn.execute(
            "UPDATE session SET status = 'open', closed_at = NULL WHERE id = ?",
            (session_id,),
        )
        conn.commit()


def delete_session(db_path: str, session_id: int) -> None:
    """Delete a session and all its queries.

    Raises ``ValueError`` if the session doesn't exist.
    """
    with _connect(db_path) as conn:
        row = conn.execute("SELECT id FROM session WHERE id = ?", (session_id,)).fetchone()
        if row is None:
            raise ValueError(f"Session {session_id} not found")
        conn.execute("DELETE FROM session_query WHERE session_id = ?", (session_id,))
        conn.execute("DELETE FROM session WHERE id = ?", (session_id,))
        conn.commit()


# ---------------------------------------------------------------------------
# Public API — recording queries
# ---------------------------------------------------------------------------


def record_query(
    db_path: str,
    session_id: int,
    *,
    question: str,
    report_path: str,
    key_claims: list[str] | None = None,
    gaps: list[str] | None = None,
    citations: list[str] | None = None,
    synthesis_preview: str = "",
    evidence_count: int = 0,
    elapsed_ms: float | None = None,
    aggregation_enabled: bool = False,
) -> int:
    """Record a completed research query in the given session.

    Returns the new query row ID.  Raises ``ValueError`` if the session
    doesn't exist or is closed.
    """
    now = _utc_now()
    with _connect(db_path) as conn:
        row = conn.execute(
            "SELECT status FROM session WHERE id = ?", (session_id,),
        ).fetchone()
        if row is None:
            raise ValueError(f"Session {session_id} not found")
        if row["status"] == "closed":
            raise ValueError(f"Session {session_id} is closed; reopen it first")
        conn.execute(
            """INSERT INTO session_query(
                session_id, question, report_path, created_at, elapsed_ms,
                key_claims_json, gaps_json, citations_json,
                synthesis_preview, evidence_count, aggregation_enabled
            ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                session_id,
                question,
                report_path,
                now,
                elapsed_ms,
                json.dumps(key_claims or []),
                json.dumps(gaps or []),
                json.dumps(citations or []),
                synthesis_preview[:500],  # cap at 500 chars
                evidence_count,
                int(aggregation_enabled),
            ),
        )
        conn.commit()
        return int(conn.execute("SELECT last_insert_rowid()").fetchone()[0])


# ---------------------------------------------------------------------------
# Public API — listing & retrieval
# ---------------------------------------------------------------------------


def list_sessions(
    db_path: str,
    status: str | None = None,
) -> list[SessionInfo]:
    """List sessions with summary stats.

    Parameters
    ----------
    status
        Filter by ``"open"`` or ``"closed"``.  ``None`` returns all.
    """
    init_session_db(db_path)
    with _connect(db_path) as conn:
        sql = """
            SELECT s.id, s.name, s.description, s.status,
                   s.created_at, s.closed_at,
                   COUNT(q.id) AS query_count
            FROM session s
            LEFT JOIN session_query q ON q.session_id = s.id
        """
        params: list[str] = []
        if status is not None:
            sql += " WHERE s.status = ?"
            params.append(status)
        sql += " GROUP BY s.id ORDER BY s.created_at DESC"
        rows = conn.execute(sql, params).fetchall()
        return [
            SessionInfo(
                session_id=r["id"],
                name=r["name"],
                description=r["description"],
                status=r["status"],
                created_at=r["created_at"],
                closed_at=r["closed_at"],
                query_count=r["query_count"],
            )
            for r in rows
        ]


def get_session_detail(db_path: str, session_id: int) -> SessionDetail:
    """Return full session detail including all queries.

    Raises ``ValueError`` if the session doesn't exist.
    """
    with _connect(db_path) as conn:
        srow = conn.execute(
            "SELECT id, name, description, status, created_at, closed_at FROM session WHERE id = ?",
            (session_id,),
        ).fetchone()
        if srow is None:
            raise ValueError(f"Session {session_id} not found")

        qcount = conn.execute(
            "SELECT COUNT(*) FROM session_query WHERE session_id = ?",
            (session_id,),
        ).fetchone()[0]

        info = SessionInfo(
            session_id=srow["id"],
            name=srow["name"],
            description=srow["description"],
            status=srow["status"],
            created_at=srow["created_at"],
            closed_at=srow["closed_at"],
            query_count=int(qcount),
        )

        qrows = conn.execute(
            """SELECT id, session_id, question, report_path, created_at,
                      elapsed_ms, key_claims_json, gaps_json, citations_json,
                      synthesis_preview, evidence_count, aggregation_enabled
               FROM session_query WHERE session_id = ? ORDER BY created_at""",
            (session_id,),
        ).fetchall()

        queries = [
            SessionQuery(
                query_id=q["id"],
                session_id=q["session_id"],
                question=q["question"],
                report_path=q["report_path"],
                created_at=q["created_at"],
                elapsed_ms=q["elapsed_ms"],
                key_claims=json.loads(q["key_claims_json"]),
                gaps=json.loads(q["gaps_json"]),
                citations=json.loads(q["citations_json"]),
                synthesis_preview=q["synthesis_preview"],
                evidence_count=q["evidence_count"],
                aggregation_enabled=bool(q["aggregation_enabled"]),
            )
            for q in qrows
        ]
        return SessionDetail(info=info, queries=queries)


# ---------------------------------------------------------------------------
# Public API — cross-query summary
# ---------------------------------------------------------------------------


def _extract_paper_ids_from_citations(citations: list[str]) -> list[str]:
    """Extract paper IDs from citation strings like ``(Ppaper:S1)``."""
    import re
    ids: list[str] = []
    for cit in citations:
        m = re.search(r"\(P([^:]+):", cit)
        if m:
            ids.append(m.group(1))
    return ids


def session_summary(db_path: str, session_id: int) -> SessionSummary:
    """Generate a cross-query summary for the session.

    Aggregates gaps, citations, papers, and timing across all queries.
    Raises ``ValueError`` if the session doesn't exist.
    """
    detail = get_session_detail(db_path, session_id)
    info = detail.info
    queries = detail.queries

    all_citations: list[str] = []
    all_gaps: list[str] = []
    all_claims: list[str] = []
    all_paper_ids: list[str] = []
    question_timeline: list[dict[str, str]] = []
    total_evidence = 0
    elapsed_values: list[float] = []
    agg_count = 0

    for q in queries:
        all_citations.extend(q.citations)
        all_gaps.extend(q.gaps)
        all_claims.extend(q.key_claims)
        all_paper_ids.extend(_extract_paper_ids_from_citations(q.citations))
        question_timeline.append({"question": q.question, "created_at": q.created_at})
        total_evidence += q.evidence_count
        if q.elapsed_ms is not None:
            elapsed_values.append(q.elapsed_ms)
        if q.aggregation_enabled:
            agg_count += 1

    # Find recurring gaps (appear in ≥2 queries' gap lists)
    gap_counter: Counter[str] = Counter()
    for q in queries:
        for g in set(q.gaps):  # dedupe within each query first
            gap_counter[g] += 1
    recurring_gaps = [g for g, count in gap_counter.items() if count >= 2]

    unique_papers = sorted(set(all_paper_ids))
    unique_citations = sorted(set(all_citations))
    avg_elapsed = (sum(elapsed_values) / len(elapsed_values)) if elapsed_values else None

    # Coverage metrics
    coverage: dict[str, Any] = {
        "unique_paper_count": len(unique_papers),
        "total_citations_count": len(all_citations),
        "unique_citations_count": len(unique_citations),
        "gap_count": len(all_gaps),
        "recurring_gap_count": len(recurring_gaps),
        "claim_count": len(all_claims),
    }

    return SessionSummary(
        session_id=session_id,
        session_name=info.name,
        query_count=info.query_count,
        total_evidence_items=total_evidence,
        unique_papers=unique_papers,
        total_citations=len(all_citations),
        unique_citations=unique_citations,
        all_gaps=all_gaps,
        recurring_gaps=recurring_gaps,
        all_key_claims=all_claims,
        question_timeline=question_timeline,
        avg_elapsed_ms=round(avg_elapsed, 3) if avg_elapsed is not None else None,
        aggregation_used_count=agg_count,
        coverage=coverage,
    )


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------


def render_session_markdown(detail: SessionDetail, summary: SessionSummary | None = None) -> str:
    """Render session detail (and optional summary) as Markdown."""
    info = detail.info
    lines = [
        f"# Research Session: {info.name}",
        "",
        f"**Status**: {info.status}  ",
        f"**Created**: {info.created_at}  ",
    ]
    if info.closed_at:
        lines.append(f"**Closed**: {info.closed_at}  ")
    if info.description:
        lines.append(f"**Description**: {info.description}  ")
    lines.append(f"**Queries**: {info.query_count}")
    lines.append("")

    # Question timeline
    lines.append("## Query Timeline")
    lines.append("")
    for i, q in enumerate(detail.queries, 1):
        elapsed = f" ({q.elapsed_ms:.0f} ms)" if q.elapsed_ms is not None else ""
        agg_tag = " [AGG]" if q.aggregation_enabled else ""
        lines.append(f"{i}. **{q.question}**{elapsed}{agg_tag}")
        lines.append(f"   - Report: `{q.report_path}`")
        lines.append(f"   - Evidence items: {q.evidence_count}")
        if q.key_claims:
            lines.append(f"   - Key claims: {len(q.key_claims)}")
        if q.gaps:
            lines.append(f"   - Gaps: {len(q.gaps)}")
        if q.synthesis_preview:
            preview = q.synthesis_preview[:200].replace("\n", " ")
            lines.append(f"   - Preview: {preview}...")
        lines.append("")

    # Summary section
    if summary is not None:
        lines.append("## Session Summary")
        lines.append("")
        lines.append(f"- **Total queries**: {summary.query_count}")
        lines.append(f"- **Total evidence items**: {summary.total_evidence_items}")
        lines.append(f"- **Unique papers referenced**: {len(summary.unique_papers)}")
        lines.append(f"- **Unique citations**: {len(summary.unique_citations)}")
        if summary.avg_elapsed_ms is not None:
            lines.append(f"- **Avg query time**: {summary.avg_elapsed_ms:.0f} ms")
        if summary.aggregation_used_count:
            lines.append(f"- **Aggregation used**: {summary.aggregation_used_count} queries")
        lines.append("")

        if summary.unique_papers:
            lines.append("### Papers Referenced")
            for pid in summary.unique_papers:
                lines.append(f"- {pid}")
            lines.append("")

        if summary.recurring_gaps:
            lines.append("### Recurring Gaps (across ≥2 queries)")
            for gap in summary.recurring_gaps:
                lines.append(f"- {gap}")
            lines.append("")

        if summary.all_key_claims:
            lines.append("### Accumulated Key Claims")
            for claim in summary.all_key_claims:
                lines.append(f"- {claim}")
            lines.append("")

    return "\n".join(lines)
