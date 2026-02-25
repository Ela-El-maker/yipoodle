"""Tests for Source Reliability Feedback Loop (Feature #5).

Covers:
- DB initialization and idempotency
- Feedback event recording and validation
- Reliability recomputation (single and all sources)
- Edge cases: no events, unknown source, invalid event type
- Scoring formula verification
- Trust multiplier mapping
- Source deletion
- Listing and sorting
- Feedback event retrieval
- Reliability report generation
- Markdown rendering
- Retrieval integration: source_trust_map in query_scored
- CLI parsing and command handlers
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.apps.source_reliability import (
    DEFAULT_RELIABILITY,
    DEFAULT_RELIABILITY_DB,
    EVENT_TYPES,
    FeedbackEvent,
    ReliabilityReport,
    SourceReliability,
    _compute_reliability,
    delete_source,
    get_feedback_events,
    get_reliability,
    get_source_trust_map,
    init_reliability_db,
    list_source_reliability,
    record_feedback,
    recompute_all,
    recompute_reliability,
    reliability_report,
    render_reliability_markdown,
    source_reliability_prior,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def db_path(tmp_path: Path) -> str:
    return str(tmp_path / "test_reliability.db")


@pytest.fixture()
def seeded_db(db_path: str) -> str:
    """Populate a DB with two sources and various feedback events."""
    # arxiv: mostly successful
    for _ in range(8):
        record_feedback(db_path, "arxiv", "fetch_success")
    for _ in range(2):
        record_feedback(db_path, "arxiv", "fetch_error")
    record_feedback(db_path, "arxiv", "extraction_quality", value=0.90)
    record_feedback(db_path, "arxiv", "extraction_quality", value=0.85)
    for _ in range(5):
        record_feedback(db_path, "arxiv", "claim_confirmed")
    record_feedback(db_path, "arxiv", "claim_disputed")
    for _ in range(3):
        record_feedback(db_path, "arxiv", "evidence_used")
    record_feedback(db_path, "arxiv", "user_upvote")
    recompute_reliability(db_path, "arxiv")

    # sketchy_blog: mostly failures
    for _ in range(2):
        record_feedback(db_path, "sketchy_blog", "fetch_success")
    for _ in range(8):
        record_feedback(db_path, "sketchy_blog", "fetch_error")
    record_feedback(db_path, "sketchy_blog", "extraction_quality", value=0.30)
    record_feedback(db_path, "sketchy_blog", "extraction_quality", value=0.25)
    for _ in range(2):
        record_feedback(db_path, "sketchy_blog", "claim_disputed")
    record_feedback(db_path, "sketchy_blog", "claim_stale")
    record_feedback(db_path, "sketchy_blog", "user_downvote")
    record_feedback(db_path, "sketchy_blog", "user_downvote")
    recompute_reliability(db_path, "sketchy_blog")

    return db_path


# ---------------------------------------------------------------------------
# TestInitDB
# ---------------------------------------------------------------------------


class TestInitDB:
    def test_init_creates_db(self, db_path: str) -> None:
        init_reliability_db(db_path)
        assert Path(db_path).exists()

    def test_init_idempotent(self, db_path: str) -> None:
        init_reliability_db(db_path)
        init_reliability_db(db_path)
        assert Path(db_path).exists()

    def test_init_creates_parent_dirs(self, tmp_path: Path) -> None:
        db = str(tmp_path / "a" / "b" / "reliability.db")
        init_reliability_db(db)
        assert Path(db).exists()


# ---------------------------------------------------------------------------
# TestRecordFeedback
# ---------------------------------------------------------------------------


class TestRecordFeedback:
    def test_record_returns_event_id(self, db_path: str) -> None:
        eid = record_feedback(db_path, "arxiv", "fetch_success")
        assert isinstance(eid, int)
        assert eid >= 1

    def test_record_all_event_types(self, db_path: str) -> None:
        for et in sorted(EVENT_TYPES):
            eid = record_feedback(db_path, "test-source", et, value=0.5)
            assert eid >= 1

    def test_record_invalid_event_type_raises(self, db_path: str) -> None:
        with pytest.raises(ValueError, match="Unknown event_type"):
            record_feedback(db_path, "arxiv", "invalid_event")

    def test_record_custom_value(self, db_path: str) -> None:
        record_feedback(db_path, "arxiv", "extraction_quality", value=0.77)
        events = get_feedback_events(db_path, "arxiv", limit=1)
        assert events[0].value == pytest.approx(0.77)

    def test_record_with_run_id(self, db_path: str) -> None:
        record_feedback(db_path, "arxiv", "fetch_success", run_id="run-001")
        events = get_feedback_events(db_path, "arxiv", limit=1)
        assert events[0].run_id == "run-001"

    def test_record_creates_source(self, db_path: str) -> None:
        record_feedback(db_path, "new-source", "fetch_success")
        sources = list_source_reliability(db_path)
        assert any(s.source_name == "new-source" for s in sources)


# ---------------------------------------------------------------------------
# TestRecomputeReliability
# ---------------------------------------------------------------------------


class TestRecomputeReliability:
    def test_recompute_perfect_source(self, db_path: str) -> None:
        for _ in range(10):
            record_feedback(db_path, "perfect", "fetch_success")
        record_feedback(db_path, "perfect", "extraction_quality", value=1.0)
        for _ in range(5):
            record_feedback(db_path, "perfect", "claim_confirmed")
        for _ in range(3):
            record_feedback(db_path, "perfect", "user_upvote")
        score = recompute_reliability(db_path, "perfect")
        # All components max out → score ≈ 1.0
        assert score > 0.9

    def test_recompute_bad_source(self, db_path: str) -> None:
        for _ in range(10):
            record_feedback(db_path, "bad", "fetch_error")
        record_feedback(db_path, "bad", "extraction_quality", value=0.1)
        for _ in range(5):
            record_feedback(db_path, "bad", "claim_disputed")
        for _ in range(3):
            record_feedback(db_path, "bad", "user_downvote")
        score = recompute_reliability(db_path, "bad")
        assert score < 0.2

    def test_recompute_no_events_returns_default(self, db_path: str) -> None:
        init_reliability_db(db_path)
        # Manually insert a source row with no events
        record_feedback(db_path, "empty-source", "fetch_success")
        # Delete all events to simulate "no events"
        import sqlite3
        conn = sqlite3.connect(db_path)
        conn.execute("DELETE FROM source_feedback_event WHERE source_name = 'empty-source'")
        conn.commit()
        conn.close()
        score = recompute_reliability(db_path, "empty-source")
        assert score == DEFAULT_RELIABILITY

    def test_recompute_unknown_source_raises(self, db_path: str) -> None:
        init_reliability_db(db_path)
        with pytest.raises(ValueError, match="not found"):
            recompute_reliability(db_path, "nonexistent")

    def test_recompute_updates_cached_row(self, db_path: str) -> None:
        for _ in range(5):
            record_feedback(db_path, "source-a", "fetch_success")
        record_feedback(db_path, "source-a", "extraction_quality", value=0.8)
        recompute_reliability(db_path, "source-a")
        sr = get_reliability(db_path, "source-a")
        assert sr.fetch_success_count == 5
        assert sr.reliability_score > 0.0

    def test_recompute_all_sources(self, seeded_db: str) -> None:
        scores = recompute_all(seeded_db)
        assert "arxiv" in scores
        assert "sketchy_blog" in scores
        assert scores["arxiv"] > scores["sketchy_blog"]


# ---------------------------------------------------------------------------
# TestComputeReliability (formula)
# ---------------------------------------------------------------------------


class TestComputeFormula:
    def test_perfect_inputs(self) -> None:
        score = _compute_reliability(1.0, 1.0, 1.0, 1.0)
        assert score == pytest.approx(1.0)

    def test_zero_inputs(self) -> None:
        score = _compute_reliability(0.0, 0.0, 0.0, 0.0)
        assert score == pytest.approx(0.0)

    def test_mixed_inputs(self) -> None:
        score = _compute_reliability(0.8, 0.9, 0.7, 0.6)
        expected = 0.30 * 0.8 + 0.25 * 0.9 + 0.30 * 0.7 + 0.15 * 0.6
        assert score == pytest.approx(expected)

    def test_clamped_above_one(self) -> None:
        score = _compute_reliability(2.0, 2.0, 2.0, 2.0)
        assert score == 1.0

    def test_clamped_below_zero(self) -> None:
        score = _compute_reliability(-1.0, -1.0, -1.0, -1.0)
        assert score == 0.0


# ---------------------------------------------------------------------------
# TestGetReliability
# ---------------------------------------------------------------------------


class TestGetReliability:
    def test_get_existing_source(self, seeded_db: str) -> None:
        sr = get_reliability(seeded_db, "arxiv")
        assert isinstance(sr, SourceReliability)
        assert sr.source_name == "arxiv"
        assert 0.0 <= sr.reliability_score <= 1.0

    def test_get_nonexistent_raises(self, db_path: str) -> None:
        init_reliability_db(db_path)
        with pytest.raises(ValueError, match="not found"):
            get_reliability(db_path, "nonexistent")


# ---------------------------------------------------------------------------
# TestListSourceReliability
# ---------------------------------------------------------------------------


class TestListSourceReliability:
    def test_empty_list(self, db_path: str) -> None:
        sources = list_source_reliability(db_path)
        assert sources == []

    def test_list_returns_all(self, seeded_db: str) -> None:
        sources = list_source_reliability(seeded_db)
        assert len(sources) == 2
        names = {s.source_name for s in sources}
        assert names == {"arxiv", "sketchy_blog"}

    def test_list_sorted_by_score(self, seeded_db: str) -> None:
        sources = list_source_reliability(seeded_db, sort_by="reliability_score")
        assert sources[0].reliability_score >= sources[1].reliability_score

    def test_list_sorted_by_name(self, seeded_db: str) -> None:
        sources = list_source_reliability(seeded_db, sort_by="source_name")
        # DESC sort by name: sketchy_blog > arxiv
        assert sources[0].source_name > sources[1].source_name

    def test_invalid_sort_falls_back(self, seeded_db: str) -> None:
        # Should not raise, falls back to reliability_score
        sources = list_source_reliability(seeded_db, sort_by="nonexistent_col")
        assert len(sources) == 2


# ---------------------------------------------------------------------------
# TestGetFeedbackEvents
# ---------------------------------------------------------------------------


class TestGetFeedbackEvents:
    def test_get_events(self, seeded_db: str) -> None:
        events = get_feedback_events(seeded_db, "arxiv", limit=5)
        assert len(events) <= 5
        assert all(isinstance(e, FeedbackEvent) for e in events)
        assert all(e.source_name == "arxiv" for e in events)

    def test_events_ordered_desc(self, seeded_db: str) -> None:
        events = get_feedback_events(seeded_db, "arxiv", limit=50)
        dates = [e.created_at for e in events]
        assert dates == sorted(dates, reverse=True)

    def test_events_empty_source(self, db_path: str) -> None:
        init_reliability_db(db_path)
        events = get_feedback_events(db_path, "nonexistent")
        assert events == []


# ---------------------------------------------------------------------------
# TestDeleteSource
# ---------------------------------------------------------------------------


class TestDeleteSource:
    def test_delete_removes_source(self, seeded_db: str) -> None:
        delete_source(seeded_db, "sketchy_blog")
        sources = list_source_reliability(seeded_db)
        assert all(s.source_name != "sketchy_blog" for s in sources)

    def test_delete_removes_events(self, seeded_db: str) -> None:
        delete_source(seeded_db, "sketchy_blog")
        events = get_feedback_events(seeded_db, "sketchy_blog")
        assert events == []

    def test_delete_nonexistent_raises(self, db_path: str) -> None:
        init_reliability_db(db_path)
        with pytest.raises(ValueError, match="not found"):
            delete_source(db_path, "nonexistent")


# ---------------------------------------------------------------------------
# TestSourceTrust
# ---------------------------------------------------------------------------


class TestSourceTrust:
    def test_prior_unknown_source_neutral(self, db_path: str) -> None:
        init_reliability_db(db_path)
        trust = source_reliability_prior("nonexistent", db_path)
        assert trust == 1.0

    def test_prior_perfect_source(self, db_path: str) -> None:
        for _ in range(10):
            record_feedback(db_path, "perfect", "fetch_success")
        record_feedback(db_path, "perfect", "extraction_quality", value=1.0)
        for _ in range(5):
            record_feedback(db_path, "perfect", "claim_confirmed")
        for _ in range(3):
            record_feedback(db_path, "perfect", "user_upvote")
        recompute_reliability(db_path, "perfect")
        trust = source_reliability_prior("perfect", db_path)
        assert trust > 1.1  # Should be close to 1.25

    def test_prior_bad_source(self, db_path: str) -> None:
        for _ in range(10):
            record_feedback(db_path, "bad", "fetch_error")
        record_feedback(db_path, "bad", "extraction_quality", value=0.0)
        for _ in range(5):
            record_feedback(db_path, "bad", "claim_disputed")
        for _ in range(3):
            record_feedback(db_path, "bad", "user_downvote")
        recompute_reliability(db_path, "bad")
        trust = source_reliability_prior("bad", db_path)
        assert trust < 0.7

    def test_trust_map(self, seeded_db: str) -> None:
        trust_map = get_source_trust_map(seeded_db)
        assert "arxiv" in trust_map
        assert "sketchy_blog" in trust_map
        assert trust_map["arxiv"] > trust_map["sketchy_blog"]
        for v in trust_map.values():
            assert 0.5 <= v <= 1.25


# ---------------------------------------------------------------------------
# TestReliabilityReport
# ---------------------------------------------------------------------------


class TestReliabilityReport:
    def test_empty_report(self, db_path: str) -> None:
        report = reliability_report(db_path)
        assert isinstance(report, ReliabilityReport)
        assert report.source_count == 0
        assert report.sources == []
        assert report.highest is None
        assert report.lowest is None

    def test_report_with_sources(self, seeded_db: str) -> None:
        report = reliability_report(seeded_db)
        assert report.source_count == 2
        assert report.highest is not None
        assert report.lowest is not None
        assert report.global_avg_reliability > 0

    def test_report_highest_lowest(self, seeded_db: str) -> None:
        report = reliability_report(seeded_db)
        assert report.highest == "arxiv"
        assert report.lowest == "sketchy_blog"


# ---------------------------------------------------------------------------
# TestRenderMarkdown
# ---------------------------------------------------------------------------


class TestRenderMarkdown:
    def test_renders_header(self, seeded_db: str) -> None:
        report = reliability_report(seeded_db)
        md = render_reliability_markdown(report)
        assert "# Source Reliability Report" in md
        assert "Sources tracked" in md

    def test_renders_table(self, seeded_db: str) -> None:
        report = reliability_report(seeded_db)
        md = render_reliability_markdown(report)
        assert "| Source |" in md
        assert "arxiv" in md
        assert "sketchy_blog" in md

    def test_renders_formula(self, seeded_db: str) -> None:
        report = reliability_report(seeded_db)
        md = render_reliability_markdown(report)
        assert "Reliability formula" in md

    def test_empty_report_renders(self, db_path: str) -> None:
        report = reliability_report(db_path)
        md = render_reliability_markdown(report)
        assert "Sources tracked" in md
        assert "0" in md


# ---------------------------------------------------------------------------
# TestRetrievalIntegration
# ---------------------------------------------------------------------------


class TestRetrievalIntegration:
    """Test that source_trust_map is applied in retrieval scoring."""

    def test_trust_map_boosts_trusted(self) -> None:
        from src.core.schemas import SnippetRecord
        from src.apps.retrieval import SimpleBM25Index

        snippets = [
            SnippetRecord(
                snippet_id="s1", paper_id="p1", section="method",
                text="transformer attention mechanism for natural language",
                page_hint=1, token_count=6, paper_year=2023,
                paper_venue="trusted", citation_count=10,
                extraction_quality_score=1.0, extraction_quality_band="good",
                extraction_source="native",
            ),
            SnippetRecord(
                snippet_id="s2", paper_id="p2", section="method",
                text="transformer attention mechanism for natural language",
                page_hint=1, token_count=6, paper_year=2023,
                paper_venue="untrusted", citation_count=10,
                extraction_quality_score=1.0, extraction_quality_band="good",
                extraction_source="native",
            ),
        ]
        idx = SimpleBM25Index.build(snippets)
        # Without trust map → same scores
        ranked_no_trust = idx.query_scored("transformer attention", top_k=2)
        assert len(ranked_no_trust) == 2
        score_no_trust_1 = ranked_no_trust[0][1]
        score_no_trust_2 = ranked_no_trust[1][1]
        # Scores should be identical (same text, same metadata)
        assert abs(score_no_trust_1 - score_no_trust_2) < 0.01

        # With trust map → trusted source gets boosted
        trust_map = {"trusted": 1.2, "untrusted": 0.6}
        ranked_trust = idx.query_scored("transformer attention", top_k=2, source_trust_map=trust_map)
        scores_by_venue = {r[0].paper_venue: r[1] for r in ranked_trust}
        assert scores_by_venue["trusted"] > scores_by_venue["untrusted"]

    def test_trust_map_none_is_neutral(self) -> None:
        from src.core.schemas import SnippetRecord
        from src.apps.retrieval import SimpleBM25Index

        snippets = [
            SnippetRecord(
                snippet_id="s1", paper_id="p1", section="method",
                text="deep learning convolution network",
                page_hint=1, token_count=4, paper_year=2023,
                paper_venue="arxiv", citation_count=5,
                extraction_quality_score=1.0, extraction_quality_band="good",
                extraction_source="native",
            ),
        ]
        idx = SimpleBM25Index.build(snippets)
        ranked_no = idx.query_scored("deep learning", top_k=2, source_trust_map=None)
        ranked_empty = idx.query_scored("deep learning", top_k=2, source_trust_map={})
        # Both should return the same score
        assert ranked_no[0][1] == pytest.approx(ranked_empty[0][1])

    def test_trust_map_unrecognized_venue_is_neutral(self) -> None:
        from src.core.schemas import SnippetRecord
        from src.apps.retrieval import SimpleBM25Index

        snippets = [
            SnippetRecord(
                snippet_id="s1", paper_id="p1", section="method",
                text="recurrent neural network language",
                page_hint=1, token_count=4, paper_year=2023,
                paper_venue="unknown_venue", citation_count=5,
                extraction_quality_score=1.0, extraction_quality_band="good",
                extraction_source="native",
            ),
        ]
        idx = SimpleBM25Index.build(snippets)
        trust_map = {"arxiv": 1.2}  # unknown_venue not in map
        ranked = idx.query_scored("recurrent neural", top_k=2, source_trust_map=trust_map)
        ranked_none = idx.query_scored("recurrent neural", top_k=2, source_trust_map=None)
        # Score shouldn't change since venue not in map
        assert ranked[0][1] == pytest.approx(ranked_none[0][1])


# ---------------------------------------------------------------------------
# TestCLIParsing
# ---------------------------------------------------------------------------


class TestCLIParsing:
    def test_reliability_list_parse(self) -> None:
        from src.cli import build_parser
        p = build_parser()
        args = p.parse_args(["reliability-list", "--sort-by", "total_events", "--format", "json"])
        assert args.sort_by == "total_events"
        assert args.format == "json"

    def test_reliability_show_parse(self) -> None:
        from src.cli import build_parser
        p = build_parser()
        args = p.parse_args(["reliability-show", "--source", "arxiv", "--events", "10"])
        assert args.source == "arxiv"
        assert args.events == 10

    def test_reliability_record_parse(self) -> None:
        from src.cli import build_parser
        p = build_parser()
        args = p.parse_args(["reliability-record", "--source", "arxiv", "--event", "fetch_success", "--value", "1.0"])
        assert args.source == "arxiv"
        assert args.event == "fetch_success"
        assert args.value == 1.0

    def test_reliability_recompute_parse(self) -> None:
        from src.cli import build_parser
        p = build_parser()
        args = p.parse_args(["reliability-recompute", "--source", "arxiv"])
        assert args.source == "arxiv"

    def test_reliability_recompute_all_parse(self) -> None:
        from src.cli import build_parser
        p = build_parser()
        args = p.parse_args(["reliability-recompute"])
        assert args.source is None

    def test_reliability_report_parse(self) -> None:
        from src.cli import build_parser
        p = build_parser()
        args = p.parse_args(["reliability-report", "--format", "json", "--out", "r.json"])
        assert args.format == "json"
        assert args.out == "r.json"

    def test_reliability_delete_parse(self) -> None:
        from src.cli import build_parser
        p = build_parser()
        args = p.parse_args(["reliability-delete", "--source", "arxiv"])
        assert args.source == "arxiv"


# ---------------------------------------------------------------------------
# TestCLIHandlers
# ---------------------------------------------------------------------------


class TestCLIHandlers:
    def test_cmd_reliability_record(self, db_path: str, capsys) -> None:
        from src.cli import cmd_reliability_record
        import argparse
        args = argparse.Namespace(reliability_db=db_path, source="arxiv",
                                   event="fetch_success", value=1.0, run_id=None)
        cmd_reliability_record(args)
        out = json.loads(capsys.readouterr().out)
        assert out["source"] == "arxiv"
        assert out["event_type"] == "fetch_success"
        assert "event_id" in out

    def test_cmd_reliability_list_json(self, seeded_db: str, capsys) -> None:
        from src.cli import cmd_reliability_list
        import argparse
        args = argparse.Namespace(reliability_db=seeded_db, sort_by="reliability_score", format="json")
        cmd_reliability_list(args)
        out = json.loads(capsys.readouterr().out)
        assert len(out) == 2
        assert out[0]["source_name"] in {"arxiv", "sketchy_blog"}

    def test_cmd_reliability_list_table(self, seeded_db: str, capsys) -> None:
        from src.cli import cmd_reliability_list
        import argparse
        args = argparse.Namespace(reliability_db=seeded_db, sort_by="reliability_score", format="table")
        cmd_reliability_list(args)
        out = capsys.readouterr().out
        assert "arxiv" in out
        assert "Score" in out

    def test_cmd_reliability_show_json(self, seeded_db: str, capsys) -> None:
        from src.cli import cmd_reliability_show
        import argparse
        args = argparse.Namespace(reliability_db=seeded_db, source="arxiv", events=5, format="json")
        cmd_reliability_show(args)
        out = json.loads(capsys.readouterr().out)
        assert out["source"]["source_name"] == "arxiv"
        assert "recent_events" in out

    def test_cmd_reliability_show_text(self, seeded_db: str, capsys) -> None:
        from src.cli import cmd_reliability_show
        import argparse
        args = argparse.Namespace(reliability_db=seeded_db, source="arxiv", events=5, format="text")
        cmd_reliability_show(args)
        out = capsys.readouterr().out
        assert "Source: arxiv" in out
        assert "Reliability score:" in out

    def test_cmd_reliability_recompute_one(self, seeded_db: str, capsys) -> None:
        from src.cli import cmd_reliability_recompute
        import argparse
        args = argparse.Namespace(reliability_db=seeded_db, source="arxiv")
        cmd_reliability_recompute(args)
        out = json.loads(capsys.readouterr().out)
        assert out["source"] == "arxiv"
        assert "reliability_score" in out

    def test_cmd_reliability_recompute_all(self, seeded_db: str, capsys) -> None:
        from src.cli import cmd_reliability_recompute
        import argparse
        args = argparse.Namespace(reliability_db=seeded_db, source=None)
        cmd_reliability_recompute(args)
        out = json.loads(capsys.readouterr().out)
        assert "arxiv" in out
        assert "sketchy_blog" in out

    def test_cmd_reliability_report_json(self, seeded_db: str, capsys) -> None:
        from src.cli import cmd_reliability_report
        import argparse
        args = argparse.Namespace(reliability_db=seeded_db, format="json", out=None)
        cmd_reliability_report(args)
        out = json.loads(capsys.readouterr().out)
        assert out["source_count"] == 2

    def test_cmd_reliability_report_writes_file(self, seeded_db: str, tmp_path: Path, capsys) -> None:
        from src.cli import cmd_reliability_report
        import argparse
        out_file = str(tmp_path / "reliability_report.md")
        args = argparse.Namespace(reliability_db=seeded_db, format="markdown", out=out_file)
        cmd_reliability_report(args)
        assert Path(out_file).exists()
        content = Path(out_file).read_text(encoding="utf-8")
        assert "Source Reliability Report" in content

    def test_cmd_reliability_delete(self, seeded_db: str, capsys) -> None:
        from src.cli import cmd_reliability_delete
        import argparse
        args = argparse.Namespace(reliability_db=seeded_db, source="sketchy_blog")
        cmd_reliability_delete(args)
        out = json.loads(capsys.readouterr().out)
        assert out["deleted"] is True
        assert out["source"] == "sketchy_blog"

    def test_cmd_reliability_list_empty(self, db_path: str, capsys) -> None:
        from src.cli import cmd_reliability_list
        import argparse
        args = argparse.Namespace(reliability_db=db_path, sort_by="reliability_score", format="table")
        cmd_reliability_list(args)
        out = capsys.readouterr().out
        assert "No sources tracked" in out


# ---------------------------------------------------------------------------
# TestEdgeCases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_mixed_events_partial_components(self, db_path: str) -> None:
        """Source with only fetch events (no quality, no claims)."""
        for _ in range(10):
            record_feedback(db_path, "fetch-only", "fetch_success")
        score = recompute_reliability(db_path, "fetch-only")
        # fetch_success_rate=1.0, avg_quality defaults 0.5, evidence_support defaults 0.5, recency defaults 0.5
        expected = 0.30 * 1.0 + 0.25 * 0.5 + 0.30 * 0.5 + 0.15 * 0.5
        assert score == pytest.approx(expected)

    def test_source_with_only_votes(self, db_path: str) -> None:
        """Source with only user votes."""
        for _ in range(3):
            record_feedback(db_path, "voted", "user_upvote")
        record_feedback(db_path, "voted", "user_downvote")
        score = recompute_reliability(db_path, "voted")
        assert 0.0 < score < 1.0

    def test_record_then_delete_then_recreate(self, db_path: str) -> None:
        """After deletion, can re-record events for same source."""
        record_feedback(db_path, "ephemeral", "fetch_success")
        recompute_reliability(db_path, "ephemeral")
        delete_source(db_path, "ephemeral")
        record_feedback(db_path, "ephemeral", "fetch_error")
        recompute_reliability(db_path, "ephemeral")
        sr = get_reliability(db_path, "ephemeral")
        assert sr.total_events == 1
        assert sr.fetch_error_count == 1

    def test_default_reliability_db_constant(self) -> None:
        assert DEFAULT_RELIABILITY_DB == "data/reliability/source_reliability.db"

    def test_event_types_complete(self) -> None:
        assert "fetch_success" in EVENT_TYPES
        assert "fetch_error" in EVENT_TYPES
        assert "extraction_quality" in EVENT_TYPES
        assert "evidence_used" in EVENT_TYPES
        assert "claim_confirmed" in EVENT_TYPES
        assert "claim_disputed" in EVENT_TYPES
        assert "claim_stale" in EVENT_TYPES
        assert "user_upvote" in EVENT_TYPES
        assert "user_downvote" in EVENT_TYPES
        assert len(EVENT_TYPES) == 9
