"""Tests for Research Session Tracking (Feature #4).

Covers:
- Session CRUD: create, close, reopen, delete
- Query recording and retrieval
- Session listing with filters
- Cross-query session summary
- Markdown rendering
- Error handling: duplicate names, closed sessions, missing sessions
- CLI integration: session subcommands and --session-id on research
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.apps.session_store import (
    SessionDetail,
    SessionInfo,
    close_session,
    create_session,
    delete_session,
    get_session_detail,
    init_session_db,
    list_sessions,
    record_query,
    render_session_markdown,
    reopen_session,
    session_summary,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def db_path(tmp_path: Path) -> str:
    return str(tmp_path / "test_sessions.db")


@pytest.fixture()
def session_with_queries(db_path: str) -> int:
    """Create a session with 3 recorded queries."""
    sid = create_session(db_path, "attention-survey", description="Survey of attention mechanisms")
    record_query(
        db_path, sid,
        question="How does attention scale?",
        report_path="runs/research_reports/q1.md",
        key_claims=["Quadratic complexity O(n^2)", "Self-attention is dominant"],
        gaps=["No analysis of linear alternatives"],
        citations=["(Pp1:S1)", "(Pp2:S1)"],
        synthesis_preview="Attention has quadratic complexity...",
        evidence_count=5,
        elapsed_ms=150.5,
        aggregation_enabled=True,
    )
    record_query(
        db_path, sid,
        question="What are linear attention alternatives?",
        report_path="runs/research_reports/q2.md",
        key_claims=["Linear attention via kernel methods", "Performer uses FAVOR+"],
        gaps=["No analysis of linear alternatives", "Missing hardware benchmarks"],
        citations=["(Pp3:S1)", "(Pp4:S1)"],
        synthesis_preview="Several linear attention variants exist...",
        evidence_count=8,
        elapsed_ms=200.3,
    )
    record_query(
        db_path, sid,
        question="How does sparse attention compare?",
        report_path="runs/research_reports/q3.md",
        key_claims=["Longformer uses sliding window"],
        gaps=["Missing hardware benchmarks"],
        citations=["(Pp5:S1)", "(Pp1:S1)"],
        synthesis_preview="Sparse attention patterns reduce cost...",
        evidence_count=3,
        elapsed_ms=120.0,
    )
    return sid


# ---------------------------------------------------------------------------
# TestSessionCreate
# ---------------------------------------------------------------------------


class TestSessionCreate:
    def test_create_returns_id(self, db_path: str) -> None:
        sid = create_session(db_path, "my-session")
        assert isinstance(sid, int)
        assert sid >= 1

    def test_create_with_description(self, db_path: str) -> None:
        sid = create_session(db_path, "my-session", description="A test session")
        detail = get_session_detail(db_path, sid)
        assert detail.info.description == "A test session"
        assert detail.info.status == "open"

    def test_create_duplicate_name_raises(self, db_path: str) -> None:
        create_session(db_path, "unique-name")
        with pytest.raises(ValueError, match="already exists"):
            create_session(db_path, "unique-name")

    def test_create_initializes_db(self, tmp_path: Path) -> None:
        db = str(tmp_path / "new_dir" / "sessions.db")
        sid = create_session(db, "first")
        assert sid >= 1
        assert Path(db).exists()

    def test_multiple_sessions_unique_ids(self, db_path: str) -> None:
        ids = [create_session(db_path, f"session-{i}") for i in range(5)]
        assert len(set(ids)) == 5


# ---------------------------------------------------------------------------
# TestSessionClose
# ---------------------------------------------------------------------------


class TestSessionClose:
    def test_close_session(self, db_path: str) -> None:
        sid = create_session(db_path, "closable")
        close_session(db_path, sid)
        detail = get_session_detail(db_path, sid)
        assert detail.info.status == "closed"
        assert detail.info.closed_at is not None

    def test_close_already_closed_raises(self, db_path: str) -> None:
        sid = create_session(db_path, "closable")
        close_session(db_path, sid)
        with pytest.raises(ValueError, match="already closed"):
            close_session(db_path, sid)

    def test_close_nonexistent_raises(self, db_path: str) -> None:
        init_session_db(db_path)
        with pytest.raises(ValueError, match="not found"):
            close_session(db_path, 999)


# ---------------------------------------------------------------------------
# TestSessionReopen
# ---------------------------------------------------------------------------


class TestSessionReopen:
    def test_reopen_closed_session(self, db_path: str) -> None:
        sid = create_session(db_path, "reopenable")
        close_session(db_path, sid)
        reopen_session(db_path, sid)
        detail = get_session_detail(db_path, sid)
        assert detail.info.status == "open"
        assert detail.info.closed_at is None

    def test_reopen_already_open_raises(self, db_path: str) -> None:
        sid = create_session(db_path, "already-open")
        with pytest.raises(ValueError, match="already open"):
            reopen_session(db_path, sid)

    def test_reopen_nonexistent_raises(self, db_path: str) -> None:
        init_session_db(db_path)
        with pytest.raises(ValueError, match="not found"):
            reopen_session(db_path, 999)


# ---------------------------------------------------------------------------
# TestSessionDelete
# ---------------------------------------------------------------------------


class TestSessionDelete:
    def test_delete_session(self, db_path: str) -> None:
        sid = create_session(db_path, "deletable")
        record_query(db_path, sid, question="q?", report_path="r.md")
        delete_session(db_path, sid)
        sessions = list_sessions(db_path)
        assert all(s.session_id != sid for s in sessions)

    def test_delete_removes_queries(self, db_path: str) -> None:
        sid = create_session(db_path, "with-queries")
        record_query(db_path, sid, question="q1?", report_path="r1.md")
        record_query(db_path, sid, question="q2?", report_path="r2.md")
        delete_session(db_path, sid)
        # Re-create to verify queries are gone
        sid2 = create_session(db_path, "with-queries")
        detail = get_session_detail(db_path, sid2)
        assert len(detail.queries) == 0

    def test_delete_nonexistent_raises(self, db_path: str) -> None:
        init_session_db(db_path)
        with pytest.raises(ValueError, match="not found"):
            delete_session(db_path, 999)


# ---------------------------------------------------------------------------
# TestRecordQuery
# ---------------------------------------------------------------------------


class TestRecordQuery:
    def test_record_returns_query_id(self, db_path: str) -> None:
        sid = create_session(db_path, "recording")
        qid = record_query(db_path, sid, question="test?", report_path="r.md")
        assert isinstance(qid, int)
        assert qid >= 1

    def test_record_stores_all_fields(self, db_path: str) -> None:
        sid = create_session(db_path, "full-fields")
        record_query(
            db_path, sid,
            question="How does X work?",
            report_path="/runs/test.md",
            key_claims=["Claim A", "Claim B"],
            gaps=["Gap 1"],
            citations=["(Pp1:S1)", "(Pp2:S2)"],
            synthesis_preview="X works by...",
            evidence_count=10,
            elapsed_ms=99.5,
            aggregation_enabled=True,
        )
        detail = get_session_detail(db_path, sid)
        q = detail.queries[0]
        assert q.question == "How does X work?"
        assert q.report_path == "/runs/test.md"
        assert q.key_claims == ["Claim A", "Claim B"]
        assert q.gaps == ["Gap 1"]
        assert q.citations == ["(Pp1:S1)", "(Pp2:S2)"]
        assert q.synthesis_preview == "X works by..."
        assert q.evidence_count == 10
        assert q.elapsed_ms == 99.5
        assert q.aggregation_enabled is True

    def test_record_on_closed_session_raises(self, db_path: str) -> None:
        sid = create_session(db_path, "closed-session")
        close_session(db_path, sid)
        with pytest.raises(ValueError, match="closed"):
            record_query(db_path, sid, question="q?", report_path="r.md")

    def test_record_on_missing_session_raises(self, db_path: str) -> None:
        init_session_db(db_path)
        with pytest.raises(ValueError, match="not found"):
            record_query(db_path, 999, question="q?", report_path="r.md")

    def test_synthesis_preview_capped(self, db_path: str) -> None:
        sid = create_session(db_path, "long-preview")
        long_text = "A" * 1000
        record_query(
            db_path, sid,
            question="q?", report_path="r.md",
            synthesis_preview=long_text,
        )
        detail = get_session_detail(db_path, sid)
        assert len(detail.queries[0].synthesis_preview) <= 500

    def test_record_multiple_queries(self, db_path: str) -> None:
        sid = create_session(db_path, "multi")
        for i in range(5):
            record_query(db_path, sid, question=f"q{i}?", report_path=f"r{i}.md")
        detail = get_session_detail(db_path, sid)
        assert len(detail.queries) == 5
        assert detail.info.query_count == 5


# ---------------------------------------------------------------------------
# TestListSessions
# ---------------------------------------------------------------------------


class TestListSessions:
    def test_empty_list(self, db_path: str) -> None:
        sessions = list_sessions(db_path)
        assert sessions == []

    def test_list_all(self, db_path: str) -> None:
        create_session(db_path, "s1")
        create_session(db_path, "s2")
        sid3 = create_session(db_path, "s3")
        close_session(db_path, sid3)
        sessions = list_sessions(db_path)
        assert len(sessions) == 3

    def test_filter_open(self, db_path: str) -> None:
        create_session(db_path, "open1")
        sid2 = create_session(db_path, "closed1")
        close_session(db_path, sid2)
        sessions = list_sessions(db_path, status="open")
        assert len(sessions) == 1
        assert sessions[0].status == "open"

    def test_filter_closed(self, db_path: str) -> None:
        create_session(db_path, "open1")
        sid2 = create_session(db_path, "closed1")
        close_session(db_path, sid2)
        sessions = list_sessions(db_path, status="closed")
        assert len(sessions) == 1
        assert sessions[0].status == "closed"

    def test_list_includes_query_count(self, db_path: str) -> None:
        sid = create_session(db_path, "with-queries")
        record_query(db_path, sid, question="q1?", report_path="r1.md")
        record_query(db_path, sid, question="q2?", report_path="r2.md")
        sessions = list_sessions(db_path)
        assert sessions[0].query_count == 2

    def test_list_returns_session_info(self, db_path: str) -> None:
        create_session(db_path, "info-check", description="Test desc")
        sessions = list_sessions(db_path)
        s = sessions[0]
        assert isinstance(s, SessionInfo)
        assert s.name == "info-check"
        assert s.description == "Test desc"
        assert s.status == "open"
        assert s.created_at


# ---------------------------------------------------------------------------
# TestGetSessionDetail
# ---------------------------------------------------------------------------


class TestGetSessionDetail:
    def test_detail_has_info_and_queries(self, db_path: str, session_with_queries: int) -> None:
        detail = get_session_detail(db_path, session_with_queries)
        assert isinstance(detail, SessionDetail)
        assert detail.info.query_count == 3
        assert len(detail.queries) == 3

    def test_queries_ordered_by_time(self, db_path: str, session_with_queries: int) -> None:
        detail = get_session_detail(db_path, session_with_queries)
        times = [q.created_at for q in detail.queries]
        assert times == sorted(times)

    def test_missing_session_raises(self, db_path: str) -> None:
        init_session_db(db_path)
        with pytest.raises(ValueError, match="not found"):
            get_session_detail(db_path, 999)


# ---------------------------------------------------------------------------
# TestSessionSummary
# ---------------------------------------------------------------------------


class TestSessionSummary:
    def test_summary_query_count(self, db_path: str, session_with_queries: int) -> None:
        s = session_summary(db_path, session_with_queries)
        assert s.query_count == 3

    def test_summary_total_evidence(self, db_path: str, session_with_queries: int) -> None:
        s = session_summary(db_path, session_with_queries)
        assert s.total_evidence_items == 16  # 5 + 8 + 3

    def test_summary_unique_papers(self, db_path: str, session_with_queries: int) -> None:
        s = session_summary(db_path, session_with_queries)
        # Citations: p1, p2, p3, p4, p5, p1 → unique: p1, p2, p3, p4, p5
        assert len(s.unique_papers) == 5

    def test_summary_total_citations(self, db_path: str, session_with_queries: int) -> None:
        s = session_summary(db_path, session_with_queries)
        assert s.total_citations == 6  # 2 + 2 + 2

    def test_summary_unique_citations(self, db_path: str, session_with_queries: int) -> None:
        s = session_summary(db_path, session_with_queries)
        # (Pp1:S1) appears in q1 and q3 → 5 unique citations
        assert len(s.unique_citations) == 5

    def test_summary_recurring_gaps(self, db_path: str, session_with_queries: int) -> None:
        s = session_summary(db_path, session_with_queries)
        # "No analysis of linear alternatives" in q1 & q2
        # "Missing hardware benchmarks" in q2 & q3
        assert len(s.recurring_gaps) == 2

    def test_summary_all_key_claims(self, db_path: str, session_with_queries: int) -> None:
        s = session_summary(db_path, session_with_queries)
        assert len(s.all_key_claims) == 5  # 2 + 2 + 1

    def test_summary_question_timeline(self, db_path: str, session_with_queries: int) -> None:
        s = session_summary(db_path, session_with_queries)
        assert len(s.question_timeline) == 3
        assert s.question_timeline[0]["question"] == "How does attention scale?"

    def test_summary_avg_elapsed(self, db_path: str, session_with_queries: int) -> None:
        s = session_summary(db_path, session_with_queries)
        assert s.avg_elapsed_ms is not None
        expected = round((150.5 + 200.3 + 120.0) / 3, 3)
        assert abs(s.avg_elapsed_ms - expected) < 0.01

    def test_summary_aggregation_count(self, db_path: str, session_with_queries: int) -> None:
        s = session_summary(db_path, session_with_queries)
        assert s.aggregation_used_count == 1  # only q1 had aggregation

    def test_summary_coverage_keys(self, db_path: str, session_with_queries: int) -> None:
        s = session_summary(db_path, session_with_queries)
        assert "unique_paper_count" in s.coverage
        assert "total_citations_count" in s.coverage
        assert "recurring_gap_count" in s.coverage
        assert "claim_count" in s.coverage

    def test_summary_empty_session(self, db_path: str) -> None:
        sid = create_session(db_path, "empty")
        s = session_summary(db_path, sid)
        assert s.query_count == 0
        assert s.total_evidence_items == 0
        assert len(s.unique_papers) == 0
        assert s.avg_elapsed_ms is None

    def test_summary_missing_session_raises(self, db_path: str) -> None:
        init_session_db(db_path)
        with pytest.raises(ValueError, match="not found"):
            session_summary(db_path, 999)


# ---------------------------------------------------------------------------
# TestRenderSessionMarkdown
# ---------------------------------------------------------------------------


class TestRenderSessionMarkdown:
    def test_renders_header(self, db_path: str, session_with_queries: int) -> None:
        detail = get_session_detail(db_path, session_with_queries)
        md = render_session_markdown(detail)
        assert "# Research Session: attention-survey" in md
        assert "**Status**: open" in md

    def test_renders_query_timeline(self, db_path: str, session_with_queries: int) -> None:
        detail = get_session_detail(db_path, session_with_queries)
        md = render_session_markdown(detail)
        assert "## Query Timeline" in md
        assert "How does attention scale?" in md
        assert "What are linear attention alternatives?" in md
        assert "How does sparse attention compare?" in md

    def test_renders_summary_section(self, db_path: str, session_with_queries: int) -> None:
        detail = get_session_detail(db_path, session_with_queries)
        s = session_summary(db_path, session_with_queries)
        md = render_session_markdown(detail, summary=s)
        assert "## Session Summary" in md
        assert "Total queries" in md
        assert "Unique papers" in md
        assert "Recurring Gaps" in md

    def test_renders_without_summary(self, db_path: str, session_with_queries: int) -> None:
        detail = get_session_detail(db_path, session_with_queries)
        md = render_session_markdown(detail, summary=None)
        assert "Session Summary" not in md

    def test_renders_closed_session(self, db_path: str) -> None:
        sid = create_session(db_path, "closed-render")
        close_session(db_path, sid)
        detail = get_session_detail(db_path, sid)
        md = render_session_markdown(detail)
        assert "**Status**: closed" in md
        assert "**Closed**:" in md

    def test_empty_session_renders(self, db_path: str) -> None:
        sid = create_session(db_path, "empty-render")
        detail = get_session_detail(db_path, sid)
        md = render_session_markdown(detail)
        assert "## Query Timeline" in md
        assert "**Queries**: 0" in md


# ---------------------------------------------------------------------------
# TestSessionCLI
# ---------------------------------------------------------------------------


class TestSessionCLI:
    def test_session_create_parse(self) -> None:
        from src.cli import build_parser
        p = build_parser()
        args = p.parse_args(["session-create", "--name", "test-session", "--description", "A test"])
        assert args.name == "test-session"
        assert args.description == "A test"

    def test_session_list_parse(self) -> None:
        from src.cli import build_parser
        p = build_parser()
        args = p.parse_args(["session-list", "--status", "open"])
        assert args.status == "open"

    def test_session_show_parse(self) -> None:
        from src.cli import build_parser
        p = build_parser()
        args = p.parse_args(["session-show", "--session-id", "1", "--format", "json"])
        assert args.session_id == 1
        assert args.format == "json"

    def test_session_summary_parse(self) -> None:
        from src.cli import build_parser
        p = build_parser()
        args = p.parse_args(["session-summary", "--session-id", "1", "--format", "json", "--out", "s.md"])
        assert args.session_id == 1
        assert args.out == "s.md"

    def test_session_close_parse(self) -> None:
        from src.cli import build_parser
        p = build_parser()
        args = p.parse_args(["session-close", "--session-id", "1"])
        assert args.session_id == 1

    def test_session_reopen_parse(self) -> None:
        from src.cli import build_parser
        p = build_parser()
        args = p.parse_args(["session-reopen", "--session-id", "1"])
        assert args.session_id == 1

    def test_session_delete_parse(self) -> None:
        from src.cli import build_parser
        p = build_parser()
        args = p.parse_args(["session-delete", "--session-id", "1"])
        assert args.session_id == 1

    def test_research_session_flags_parse(self) -> None:
        from src.cli import build_parser
        p = build_parser()
        args = p.parse_args([
            "research", "--index", "idx.json", "--question", "test?",
            "--session-id", "42", "--session-db", "/tmp/s.db",
        ])
        assert args.session_id == 42
        assert args.session_db == "/tmp/s.db"

    def test_research_session_flag_defaults_none(self) -> None:
        from src.cli import build_parser
        p = build_parser()
        args = p.parse_args(["research", "--index", "idx.json", "--question", "test?"])
        assert args.session_id is None


# ---------------------------------------------------------------------------
# TestSessionCLIHandlers
# ---------------------------------------------------------------------------


class TestSessionCLIHandlers:
    """Test CLI handler functions directly (no subprocess, no run_research)."""

    def test_cmd_session_create(self, db_path: str, capsys) -> None:
        from src.cli import cmd_session_create
        import argparse
        args = argparse.Namespace(name="cli-session", description="via CLI", session_db=db_path)
        cmd_session_create(args)
        out = json.loads(capsys.readouterr().out)
        assert out["name"] == "cli-session"
        assert out["status"] == "open"
        assert "session_id" in out

    def test_cmd_session_list(self, db_path: str, capsys) -> None:
        from src.cli import cmd_session_list
        import argparse
        create_session(db_path, "s1")
        create_session(db_path, "s2")
        args = argparse.Namespace(session_db=db_path, status="all")
        cmd_session_list(args)
        out = json.loads(capsys.readouterr().out)
        assert len(out) == 2

    def test_cmd_session_show_json(self, db_path: str, capsys, session_with_queries: int) -> None:
        from src.cli import cmd_session_show
        import argparse
        args = argparse.Namespace(session_db=db_path, session_id=session_with_queries, format="json")
        cmd_session_show(args)
        out = json.loads(capsys.readouterr().out)
        assert out["session"]["query_count"] == 3
        assert len(out["queries"]) == 3

    def test_cmd_session_show_markdown(self, db_path: str, capsys, session_with_queries: int) -> None:
        from src.cli import cmd_session_show
        import argparse
        args = argparse.Namespace(session_db=db_path, session_id=session_with_queries, format="markdown")
        cmd_session_show(args)
        out = capsys.readouterr().out
        assert "# Research Session" in out

    def test_cmd_session_summary_json(self, db_path: str, capsys, session_with_queries: int) -> None:
        from src.cli import cmd_session_summary
        import argparse
        args = argparse.Namespace(session_db=db_path, session_id=session_with_queries, format="json", out=None)
        cmd_session_summary(args)
        out = json.loads(capsys.readouterr().out)
        assert out["query_count"] == 3
        assert "recurring_gaps" in out

    def test_cmd_session_summary_writes_file(self, db_path: str, tmp_path: Path, capsys, session_with_queries: int) -> None:
        from src.cli import cmd_session_summary
        import argparse
        out_file = str(tmp_path / "summary.md")
        args = argparse.Namespace(session_db=db_path, session_id=session_with_queries, format="markdown", out=out_file)
        cmd_session_summary(args)
        assert Path(out_file).exists()
        content = Path(out_file).read_text(encoding="utf-8")
        assert "Session Summary" in content

    def test_cmd_session_close(self, db_path: str, capsys) -> None:
        from src.cli import cmd_session_close
        import argparse
        sid = create_session(db_path, "to-close")
        args = argparse.Namespace(session_db=db_path, session_id=sid)
        cmd_session_close(args)
        out = json.loads(capsys.readouterr().out)
        assert out["status"] == "closed"

    def test_cmd_session_reopen(self, db_path: str, capsys) -> None:
        from src.cli import cmd_session_reopen
        import argparse
        sid = create_session(db_path, "to-reopen")
        close_session(db_path, sid)
        args = argparse.Namespace(session_db=db_path, session_id=sid)
        cmd_session_reopen(args)
        out = json.loads(capsys.readouterr().out)
        assert out["status"] == "open"

    def test_cmd_session_delete(self, db_path: str, capsys) -> None:
        from src.cli import cmd_session_delete
        import argparse
        sid = create_session(db_path, "to-delete")
        args = argparse.Namespace(session_db=db_path, session_id=sid)
        cmd_session_delete(args)
        out = json.loads(capsys.readouterr().out)
        assert out["deleted"] is True


# ---------------------------------------------------------------------------
# TestSessionRecordingFromResearch
# ---------------------------------------------------------------------------


class TestSessionRecordingFromResearch:
    """Test the recording workflow used by cmd_research's --session-id flag."""

    def test_record_from_report_files(self, db_path: str, tmp_path: Path) -> None:
        """Simulate what cmd_research does: write report files, then record."""
        sid = create_session(db_path, "research-session")

        # Simulate report outputs
        report_data = {
            "question": "How does attention scale?",
            "synthesis": "Attention has quadratic cost...",
            "key_claims": ["Claim A"],
            "gaps": ["Gap 1"],
            "citations": ["(Pp1:S1)"],
            "shortlist": [],
            "experiments": [],
        }
        evidence_data = {
            "question": "How does attention scale?",
            "items": [{"paper_id": "p1", "snippet_id": "Pp1:S1", "score": 0.9,
                        "section": "method", "text": "Some evidence."}],
        }
        metrics_data = {"elapsed_ms_total": 250.0}

        report_path = tmp_path / "report.json"
        evidence_path = tmp_path / "report.evidence.json"
        metrics_path = tmp_path / "report.metrics.json"
        report_path.write_text(json.dumps(report_data), encoding="utf-8")
        evidence_path.write_text(json.dumps(evidence_data), encoding="utf-8")
        metrics_path.write_text(json.dumps(metrics_data), encoding="utf-8")

        # Record query
        record_query(
            db_path, sid,
            question=report_data["question"],
            report_path=str(tmp_path / "report.md"),
            key_claims=report_data["key_claims"],
            gaps=report_data["gaps"],
            citations=report_data["citations"],
            synthesis_preview=report_data["synthesis"][:500],
            evidence_count=len(evidence_data["items"]),
            elapsed_ms=metrics_data["elapsed_ms_total"],
        )

        detail = get_session_detail(db_path, sid)
        assert detail.info.query_count == 1
        q = detail.queries[0]
        assert q.question == "How does attention scale?"
        assert q.evidence_count == 1
        assert q.elapsed_ms == 250.0


# ---------------------------------------------------------------------------
# TestSessionEdgeCases
# ---------------------------------------------------------------------------


class TestSessionEdgeCases:
    def test_reopen_then_record(self, db_path: str) -> None:
        """After reopen, recording should work again."""
        sid = create_session(db_path, "reopen-record")
        record_query(db_path, sid, question="q1?", report_path="r1.md")
        close_session(db_path, sid)
        reopen_session(db_path, sid)
        record_query(db_path, sid, question="q2?", report_path="r2.md")
        detail = get_session_detail(db_path, sid)
        assert detail.info.query_count == 2

    def test_concurrent_sessions(self, db_path: str) -> None:
        """Multiple sessions can be open simultaneously."""
        s1 = create_session(db_path, "session-a")
        s2 = create_session(db_path, "session-b")
        record_query(db_path, s1, question="q1?", report_path="r1.md")
        record_query(db_path, s2, question="q2?", report_path="r2.md")
        d1 = get_session_detail(db_path, s1)
        d2 = get_session_detail(db_path, s2)
        assert d1.info.query_count == 1
        assert d2.info.query_count == 1

    def test_empty_citations_no_papers(self, db_path: str) -> None:
        """Session summary handles queries with no citations gracefully."""
        sid = create_session(db_path, "no-cites")
        record_query(db_path, sid, question="q?", report_path="r.md",
                      citations=[], key_claims=[], gaps=[])
        s = session_summary(db_path, sid)
        assert len(s.unique_papers) == 0
        assert s.total_citations == 0

    def test_no_elapsed_ms(self, db_path: str) -> None:
        """Session summary handles None elapsed_ms gracefully."""
        sid = create_session(db_path, "no-timing")
        record_query(db_path, sid, question="q?", report_path="r.md",
                      elapsed_ms=None)
        s = session_summary(db_path, sid)
        assert s.avg_elapsed_ms is None

    def test_init_session_db_idempotent(self, db_path: str) -> None:
        """Calling init_session_db multiple times is safe."""
        init_session_db(db_path)
        init_session_db(db_path)
        create_session(db_path, "after-double-init")
        assert len(list_sessions(db_path)) == 1
