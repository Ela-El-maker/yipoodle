import json

from src.apps.corpus_health import evaluate_corpus_health


def test_corpus_health_healthy_fixture() -> None:
    stats = evaluate_corpus_health(
        corpus_dir="tests/fixtures/extracted",
        min_snippets=1,
        min_avg_chars_per_paper=50,
        min_avg_chars_per_page=0,
        max_extract_error_rate=1.0,
    )
    assert stats["healthy"] is True
    assert int(stats["paper_count"]) >= 1
    assert int(stats["snippet_count"]) >= 1
    assert "warnings" in stats


def test_corpus_health_unhealthy_empty(tmp_path) -> None:
    corpus = tmp_path / "empty"
    corpus.mkdir()
    stats = evaluate_corpus_health(
        corpus_dir=str(corpus),
        min_snippets=1,
        min_avg_chars_per_paper=10,
        min_avg_chars_per_page=1,
        max_extract_error_rate=0.8,
    )
    assert stats["healthy"] is False
    assert any("snippet_count_below_threshold" in r for r in stats["reasons"])
    assert any("avg_chars_per_page_below_threshold" in r for r in stats["reasons"])
    assert stats["warnings"] == []


def test_corpus_health_uses_extract_stats(tmp_path) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    payload = {"paper": {"paper_id": "p1"}, "snippets": [{"text": "hello world"}]}
    (corpus / "p1.json").write_text(json.dumps(payload), encoding="utf-8")
    extract_stats = tmp_path / "extract.stats.json"
    extract_stats.write_text(json.dumps({"processed": 10, "created": 1, "failed_pdfs_count": 9}), encoding="utf-8")
    stats = evaluate_corpus_health(
        corpus_dir=str(corpus),
        extract_stats_path=str(extract_stats),
        min_snippets=1,
        min_avg_chars_per_paper=1,
        min_avg_chars_per_page=0,
        max_extract_error_rate=0.5,
    )
    assert stats["healthy"] is False
    assert any("extract_error_rate_above_threshold" in r for r in stats["reasons"])


def test_corpus_health_uses_page_stats_threshold(tmp_path) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    payload = {
        "paper": {"paper_id": "p1"},
        "snippets": [{"text": "x" * 120}],
        "extraction_meta": {"pages_total": 4},
    }
    (corpus / "p1.json").write_text(json.dumps(payload), encoding="utf-8")
    stats = evaluate_corpus_health(
        corpus_dir=str(corpus),
        min_snippets=1,
        min_avg_chars_per_paper=1,
        min_avg_chars_per_page=40,
        max_extract_error_rate=1.0,
    )
    assert stats["healthy"] is False
    assert float(stats["avg_chars_per_page"]) == 30.0
    assert float(stats["page_stats_coverage_pct"]) == 100.0
    assert any("avg_chars_per_page_below_threshold" in r for r in stats["reasons"])


def test_corpus_health_warns_when_page_stats_missing(tmp_path) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    payload = {"paper": {"paper_id": "p1"}, "snippets": [{"text": "hello world"}]}
    (corpus / "p1.json").write_text(json.dumps(payload), encoding="utf-8")
    stats = evaluate_corpus_health(
        corpus_dir=str(corpus),
        min_snippets=1,
        min_avg_chars_per_paper=1,
        min_avg_chars_per_page=0,
        max_extract_error_rate=1.0,
    )
    assert stats["healthy"] is True
    assert "page_stats_missing_for_corpus" in stats["warnings"]
