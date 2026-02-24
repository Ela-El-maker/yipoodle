from pathlib import Path
import json

from src.apps.extraction_quality_report import build_extraction_quality_report, write_extraction_quality_report


def test_build_extraction_quality_report(tmp_path) -> None:
    corpus = tmp_path / "extracted"
    corpus.mkdir()
    payload = {
        "paper": {"paper_id": "p1"},
        "snippets": [{"text": "sample snippet text for quality report"}],
        "extraction_meta": {
            "extractor": "pypdf",
            "quality_score": 0.61,
            "quality_band": "ok",
            "ocr_applied": True,
            "pages_total": 3,
            "empty_pages": 1,
            "empty_page_pct": 0.3333,
            "page_stats": [
                {"page": 1, "chars": 100, "empty": False},
                {"page": 2, "chars": 0, "empty": True},
                {"page": 3, "chars": 20, "empty": False},
            ],
        },
    }
    (corpus / "p1.json").write_text(json.dumps(payload), encoding="utf-8")
    report = build_extraction_quality_report(str(corpus))
    assert report["summary"]["papers"] == 1
    assert report["summary"]["ocr_papers"] == 1
    assert report["summary"]["empty_page_pct"] > 0
    assert report["papers"][0]["paper_id"] == "p1"
    assert len(report["papers"][0]["worst_pages"]) >= 1


def test_write_extraction_quality_report(tmp_path) -> None:
    corpus = tmp_path / "extracted"
    corpus.mkdir()
    payload = {"paper": {"paper_id": "p1"}, "snippets": [{"text": "abc"}], "extraction_meta": {}}
    (corpus / "p1.json").write_text(json.dumps(payload), encoding="utf-8")
    out = tmp_path / "quality.md"
    written = write_extraction_quality_report(str(corpus), str(out))
    assert written == str(out)
    assert out.exists()
    assert out.with_suffix(".json").exists()
