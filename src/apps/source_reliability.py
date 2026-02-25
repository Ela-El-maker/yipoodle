"""Source Reliability Feedback Loop (Feature #5).

Tracks per-source reliability scores via a SQLite-backed event log.
Each connector (arxiv, openalex, semanticscholar, …) accumulates
feedback events (fetch success/error, extraction quality, claim
confirmation/dispute, user up/down-votes).  A weighted formula
recomputes the reliability score which can be consumed as a prior
in the retrieval pipeline (see ``source_reliability_prior``).

Database layout
---------------
* **source_reliability** – one row per connector, caching the latest
  aggregate score plus counters.
* **source_feedback_event** – append-only event log with event type,
  numeric value, optional run_id, and timestamp.

Usage::

    from src.apps.source_reliability import (
        record_feedback, recompute_reliability,
        get_reliability, list_source_reliability,
        source_reliability_prior,
    )

    record_feedback(db, "arxiv", "fetch_success")
    record_feedback(db, "arxiv", "extraction_quality", value=0.92)
    recompute_reliability(db, "arxiv")
    trust = source_reliability_prior("arxiv", db)  # → float in [0.5, 1.25]
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_RELIABILITY_DB = "data/reliability/source_reliability.db"

EVENT_TYPES = frozenset({
    "fetch_success",
    "fetch_error",
    "extraction_quality",
    "evidence_used",
    "claim_confirmed",
    "claim_disputed",
    "claim_stale",
    "user_upvote",
    "user_downvote",
})

# Weights for the aggregate reliability formula
_W_FETCH = 0.30
_W_QUALITY = 0.25
_W_EVIDENCE = 0.30
_W_RECENCY = 0.15

# Default score for sources with no events
DEFAULT_RELIABILITY = 0.50

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class SourceReliability:
    """Cached per-source reliability row."""

    source_name: str
    reliability_score: float
    total_events: int
    fetch_success_count: int
    fetch_error_count: int
    avg_extraction_quality: float | None
    claims_confirmed: int
    claims_disputed: int
    claims_stale: int
    user_upvotes: int
    user_downvotes: int
    last_updated_at: str
    created_at: str


@dataclass
class FeedbackEvent:
    """Single feedback event."""

    event_id: int
    source_name: str
    event_type: str
    value: float
    run_id: str | None
    created_at: str


@dataclass
class ReliabilityReport:
    """Full reliability report across all sources."""

    generated_at: str
    source_count: int
    sources: list[SourceReliability]
    global_avg_reliability: float
    highest: str | None
    lowest: str | None


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_SCHEMA_SQL = """\
CREATE TABLE IF NOT EXISTS source_reliability (
    id                     INTEGER PRIMARY KEY AUTOINCREMENT,
    source_name            TEXT    NOT NULL UNIQUE,
    reliability_score      REAL    NOT NULL DEFAULT 0.5,
    total_events           INTEGER NOT NULL DEFAULT 0,
    fetch_success_count    INTEGER NOT NULL DEFAULT 0,
    fetch_error_count      INTEGER NOT NULL DEFAULT 0,
    avg_extraction_quality REAL,
    claims_confirmed       INTEGER NOT NULL DEFAULT 0,
    claims_disputed        INTEGER NOT NULL DEFAULT 0,
    claims_stale           INTEGER NOT NULL DEFAULT 0,
    user_upvotes           INTEGER NOT NULL DEFAULT 0,
    user_downvotes         INTEGER NOT NULL DEFAULT 0,
    last_updated_at        TEXT    NOT NULL,
    created_at             TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS source_feedback_event (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    source_name TEXT    NOT NULL,
    event_type  TEXT    NOT NULL,
    value       REAL    NOT NULL DEFAULT 1.0,
    run_id      TEXT,
    created_at  TEXT    NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_sfe_source
    ON source_feedback_event(source_name);
CREATE INDEX IF NOT EXISTS idx_sfe_created
    ON source_feedback_event(created_at);
"""

# ---------------------------------------------------------------------------
# DB helpers (mirrors session_store / kb_store pattern)
# ---------------------------------------------------------------------------


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def _connect(db_path: str) -> Iterator[sqlite3.Connection]:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_reliability_db(db_path: str = DEFAULT_RELIABILITY_DB) -> None:
    """Create tables if they don't exist (idempotent)."""
    with _connect(db_path) as conn:
        conn.executescript(_SCHEMA_SQL)


# ---------------------------------------------------------------------------
# CRUD
# ---------------------------------------------------------------------------


def record_feedback(
    db_path: str,
    source_name: str,
    event_type: str,
    value: float = 1.0,
    run_id: str | None = None,
) -> int:
    """Append a feedback event for *source_name*.

    Returns the new event id.  Raises ``ValueError`` for unknown event types.
    """
    if event_type not in EVENT_TYPES:
        raise ValueError(
            f"Unknown event_type {event_type!r}. "
            f"Must be one of: {', '.join(sorted(EVENT_TYPES))}"
        )
    now = _utc_now()
    with _connect(db_path) as conn:
        conn.executescript(_SCHEMA_SQL)
        cur = conn.execute(
            "INSERT INTO source_feedback_event (source_name, event_type, value, run_id, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (source_name, event_type, value, run_id, now),
        )
        # Ensure source_reliability row exists
        conn.execute(
            "INSERT OR IGNORE INTO source_reliability "
            "(source_name, reliability_score, total_events, fetch_success_count, "
            "fetch_error_count, claims_confirmed, claims_disputed, claims_stale, "
            "user_upvotes, user_downvotes, last_updated_at, created_at) "
            "VALUES (?, ?, 0, 0, 0, 0, 0, 0, 0, 0, ?, ?)",
            (source_name, DEFAULT_RELIABILITY, now, now),
        )
        return cur.lastrowid  # type: ignore[return-value]


def _compute_reliability(
    fetch_success_rate: float,
    avg_quality: float,
    evidence_support_rate: float,
    recency_factor: float,
) -> float:
    """Weighted reliability score clamped to [0, 1]."""
    raw = (
        _W_FETCH * fetch_success_rate
        + _W_QUALITY * avg_quality
        + _W_EVIDENCE * evidence_support_rate
        + _W_RECENCY * recency_factor
    )
    return min(1.0, max(0.0, raw))


def recompute_reliability(
    db_path: str,
    source_name: str,
) -> float:
    """Recompute and persist the reliability score for *source_name*.

    Returns the new score.  Raises ``ValueError`` if the source doesn't exist.
    """
    now = _utc_now()
    with _connect(db_path) as conn:
        conn.executescript(_SCHEMA_SQL)
        # Gather event counts
        rows = conn.execute(
            "SELECT event_type, COUNT(*) as cnt, AVG(value) as avg_val "
            "FROM source_feedback_event WHERE source_name = ? GROUP BY event_type",
            (source_name,),
        ).fetchall()
        if not rows:
            row = conn.execute(
                "SELECT id FROM source_reliability WHERE source_name = ?",
                (source_name,),
            ).fetchone()
            if row is None:
                raise ValueError(f"Source {source_name!r} not found")
            return DEFAULT_RELIABILITY

        counts: dict[str, int] = {}
        avgs: dict[str, float] = {}
        for r in rows:
            counts[r["event_type"]] = int(r["cnt"])
            avgs[r["event_type"]] = float(r["avg_val"])

        total_events = sum(counts.values())

        # Fetch success rate
        fetch_ok = counts.get("fetch_success", 0)
        fetch_err = counts.get("fetch_error", 0)
        fetch_total = fetch_ok + fetch_err
        fetch_success_rate = (fetch_ok / fetch_total) if fetch_total > 0 else 0.5

        # Extraction quality
        avg_quality = avgs.get("extraction_quality", 0.5)

        # Evidence support rate (confirmed / (confirmed + disputed + stale))
        confirmed = counts.get("claim_confirmed", 0)
        disputed = counts.get("claim_disputed", 0)
        stale = counts.get("claim_stale", 0)
        claims_total = confirmed + disputed + stale
        evidence_support_rate = (confirmed / claims_total) if claims_total > 0 else 0.5

        # Recency factor: user votes + evidence_used as positive signal
        upvotes = counts.get("user_upvote", 0)
        downvotes = counts.get("user_downvote", 0)
        evidence_used = counts.get("evidence_used", 0)
        vote_signal = upvotes + evidence_used
        vote_total = vote_signal + downvotes
        recency_factor = (vote_signal / vote_total) if vote_total > 0 else 0.5

        score = _compute_reliability(
            fetch_success_rate, avg_quality, evidence_support_rate, recency_factor,
        )

        conn.execute(
            "UPDATE source_reliability SET "
            "reliability_score = ?, total_events = ?, "
            "fetch_success_count = ?, fetch_error_count = ?, "
            "avg_extraction_quality = ?, "
            "claims_confirmed = ?, claims_disputed = ?, claims_stale = ?, "
            "user_upvotes = ?, user_downvotes = ?, last_updated_at = ? "
            "WHERE source_name = ?",
            (
                score, total_events,
                fetch_ok, fetch_err,
                avg_quality,
                confirmed, disputed, stale,
                upvotes, downvotes, now,
                source_name,
            ),
        )
        return score


def recompute_all(db_path: str) -> dict[str, float]:
    """Recompute reliability for every known source.  Returns ``{name: score}``."""
    with _connect(db_path) as conn:
        conn.executescript(_SCHEMA_SQL)
        names = [
            r["source_name"]
            for r in conn.execute(
                "SELECT DISTINCT source_name FROM source_reliability ORDER BY source_name"
            ).fetchall()
        ]
    return {name: recompute_reliability(db_path, name) for name in names}


def get_reliability(db_path: str, source_name: str) -> SourceReliability:
    """Return the cached reliability row for *source_name*.

    Raises ``ValueError`` if the source doesn't exist.
    """
    with _connect(db_path) as conn:
        conn.executescript(_SCHEMA_SQL)
        row = conn.execute(
            "SELECT * FROM source_reliability WHERE source_name = ?",
            (source_name,),
        ).fetchone()
        if row is None:
            raise ValueError(f"Source {source_name!r} not found")
        return SourceReliability(
            source_name=row["source_name"],
            reliability_score=row["reliability_score"],
            total_events=row["total_events"],
            fetch_success_count=row["fetch_success_count"],
            fetch_error_count=row["fetch_error_count"],
            avg_extraction_quality=row["avg_extraction_quality"],
            claims_confirmed=row["claims_confirmed"],
            claims_disputed=row["claims_disputed"],
            claims_stale=row["claims_stale"],
            user_upvotes=row["user_upvotes"],
            user_downvotes=row["user_downvotes"],
            last_updated_at=row["last_updated_at"],
            created_at=row["created_at"],
        )


def list_source_reliability(
    db_path: str,
    sort_by: str = "reliability_score",
) -> list[SourceReliability]:
    """Return all source reliability rows, sorted descending."""
    valid_sorts = {
        "reliability_score", "source_name", "total_events",
        "created_at", "last_updated_at",
    }
    if sort_by not in valid_sorts:
        sort_by = "reliability_score"
    with _connect(db_path) as conn:
        conn.executescript(_SCHEMA_SQL)
        rows = conn.execute(
            f"SELECT * FROM source_reliability ORDER BY {sort_by} DESC",
        ).fetchall()
        return [
            SourceReliability(
                source_name=r["source_name"],
                reliability_score=r["reliability_score"],
                total_events=r["total_events"],
                fetch_success_count=r["fetch_success_count"],
                fetch_error_count=r["fetch_error_count"],
                avg_extraction_quality=r["avg_extraction_quality"],
                claims_confirmed=r["claims_confirmed"],
                claims_disputed=r["claims_disputed"],
                claims_stale=r["claims_stale"],
                user_upvotes=r["user_upvotes"],
                user_downvotes=r["user_downvotes"],
                last_updated_at=r["last_updated_at"],
                created_at=r["created_at"],
            )
            for r in rows
        ]


def get_feedback_events(
    db_path: str,
    source_name: str,
    limit: int = 50,
) -> list[FeedbackEvent]:
    """Return the most recent feedback events for *source_name*."""
    with _connect(db_path) as conn:
        conn.executescript(_SCHEMA_SQL)
        rows = conn.execute(
            "SELECT * FROM source_feedback_event "
            "WHERE source_name = ? ORDER BY created_at DESC LIMIT ?",
            (source_name, limit),
        ).fetchall()
        return [
            FeedbackEvent(
                event_id=r["id"],
                source_name=r["source_name"],
                event_type=r["event_type"],
                value=r["value"],
                run_id=r["run_id"],
                created_at=r["created_at"],
            )
            for r in rows
        ]


def delete_source(db_path: str, source_name: str) -> None:
    """Delete a source and all its feedback events.

    Raises ``ValueError`` if the source doesn't exist.
    """
    with _connect(db_path) as conn:
        conn.executescript(_SCHEMA_SQL)
        row = conn.execute(
            "SELECT id FROM source_reliability WHERE source_name = ?",
            (source_name,),
        ).fetchone()
        if row is None:
            raise ValueError(f"Source {source_name!r} not found")
        conn.execute(
            "DELETE FROM source_feedback_event WHERE source_name = ?",
            (source_name,),
        )
        conn.execute(
            "DELETE FROM source_reliability WHERE source_name = ?",
            (source_name,),
        )


# ---------------------------------------------------------------------------
# Trust multiplier for retrieval integration
# ---------------------------------------------------------------------------


def source_reliability_prior(
    source_name: str,
    db_path: str = DEFAULT_RELIABILITY_DB,
) -> float:
    """Map a source's reliability score to a retrieval multiplier in [0.5, 1.25].

    If the source has no reliability data, returns 1.0 (neutral).
    The mapping is linear:
      - score 0.0  → 0.50
      - score 0.5  → 0.875
      - score 1.0  → 1.25
    """
    try:
        sr = get_reliability(db_path, source_name)
    except ValueError:
        return 1.0
    # Linear map [0, 1] → [0.5, 1.25]
    return 0.5 + 0.75 * sr.reliability_score


def get_source_trust_map(db_path: str = DEFAULT_RELIABILITY_DB) -> dict[str, float]:
    """Build a {source_name: trust_multiplier} dict for all known sources.

    Useful for passing into retrieval as a bulk lookup so that individual
    per-snippet DB queries are avoided.
    """
    sources = list_source_reliability(db_path)
    return {s.source_name: 0.5 + 0.75 * s.reliability_score for s in sources}


# ---------------------------------------------------------------------------
# Reliability report
# ---------------------------------------------------------------------------


def reliability_report(db_path: str) -> ReliabilityReport:
    """Generate a comprehensive reliability report across all sources."""
    sources = list_source_reliability(db_path)
    if not sources:
        return ReliabilityReport(
            generated_at=_utc_now(),
            source_count=0,
            sources=[],
            global_avg_reliability=0.0,
            highest=None,
            lowest=None,
        )
    avg = sum(s.reliability_score for s in sources) / len(sources)
    best = max(sources, key=lambda s: s.reliability_score)
    worst = min(sources, key=lambda s: s.reliability_score)
    return ReliabilityReport(
        generated_at=_utc_now(),
        source_count=len(sources),
        sources=sources,
        global_avg_reliability=round(avg, 4),
        highest=best.source_name,
        lowest=worst.source_name,
    )


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------


def render_reliability_markdown(report: ReliabilityReport) -> str:
    """Render a reliability report as Markdown."""
    lines: list[str] = [
        "# Source Reliability Report",
        "",
        f"**Generated**: {report.generated_at}",
        f"**Sources tracked**: {report.source_count}",
        f"**Global average reliability**: {report.global_avg_reliability:.4f}",
    ]
    if report.highest:
        lines.append(f"**Most reliable**: {report.highest}")
    if report.lowest:
        lines.append(f"**Least reliable**: {report.lowest}")
    lines.append("")

    if report.sources:
        lines.append("## Per-Source Breakdown")
        lines.append("")
        lines.append("| Source | Score | Events | Fetch OK | Fetch Err | Avg Quality | Confirmed | Disputed | Stale | ↑ | ↓ |")
        lines.append("|--------|-------|--------|----------|-----------|-------------|-----------|----------|-------|---|---|")
        for s in report.sources:
            aq = f"{s.avg_extraction_quality:.3f}" if s.avg_extraction_quality is not None else "—"
            lines.append(
                f"| {s.source_name} | {s.reliability_score:.4f} | {s.total_events} | "
                f"{s.fetch_success_count} | {s.fetch_error_count} | {aq} | "
                f"{s.claims_confirmed} | {s.claims_disputed} | {s.claims_stale} | "
                f"{s.user_upvotes} | {s.user_downvotes} |"
            )
        lines.append("")

    lines.append("---")
    lines.append("*Reliability formula: "
                 f"fetch_rate×{_W_FETCH} + quality×{_W_QUALITY} + "
                 f"evidence_support×{_W_EVIDENCE} + recency×{_W_RECENCY}*")
    return "\n".join(lines) + "\n"
