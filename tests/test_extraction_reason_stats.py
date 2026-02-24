from pathlib import Path

from src.apps import evidence_extract as ee


def test_extract_reason_stats_counts(tmp_path, monkeypatch) -> None:
    papers = tmp_path / "papers"
    out = tmp_path / "extracted"
    papers.mkdir()
    for n in ["a", "b", "c", "d"]:
        (papers / f"{n}.pdf").write_bytes(b"%PDF-1.4")

    responses = {
        str(papers / "a.pdf"): ("", "pypdf", None, False),  # empty_text
        str(papers / "b.pdf"): ("tiny", "pypdf", None, False),  # low_text
        str(papers / "c.pdf"): (None, None, "all_extractors_failed", False),  # extract error
        str(papers / "d.pdf"): ("good enough content " * 20, "pdfminer", None, False),  # created
    }

    monkeypatch.setattr(ee, "_extract_text_with_fallback", lambda p, two_column_mode="auto", **kwargs: responses[p])
    monkeypatch.setattr(
        ee,
        "extract_snippets",
        lambda paper, text, extraction_quality_score=None, extraction_quality_band=None, extraction_source=None: [
            ee.SnippetRecord(
                snippet_id=f"{paper.paper_id}:S1",
                paper_id=paper.paper_id,
                section="body",
                text=text,
                token_count=3,
                extraction_quality_score=extraction_quality_score,
                extraction_quality_band=extraction_quality_band,
                extraction_source=extraction_source,
            )
        ],
    )

    stats = ee.extract_from_papers_dir(str(papers), str(out), min_text_chars=20)
    assert stats["processed"] == 4
    assert stats["created"] == 1
    assert stats["extract_errors"] == 1
    assert stats["empty_text_skipped"] == 1
    assert stats["low_text_skipped"] == 1
    assert stats["failed_pdfs_count"] == 3
    assert stats["failed_reason_counts"]["empty_text"] == 1
    assert stats["failed_reason_counts"]["low_text"] == 1
    assert stats["failed_reason_counts"]["all_extractors_failed"] == 1
    assert stats["extractor_used_counts"]["pdfminer"] == 1
    assert len(stats["failed_pdfs"]) == 3
    assert stats["layout_engine"] == "shadow"
    assert "layout_region_counts" in stats
    assert "layout_shadow_diff_rate" in stats
