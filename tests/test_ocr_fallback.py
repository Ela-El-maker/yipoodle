from src.apps import evidence_extract as ee


def test_ocr_triggered_on_low_text_and_succeeds(tmp_path, monkeypatch) -> None:
    papers = tmp_path / "papers"
    out = tmp_path / "extracted"
    papers.mkdir()
    (papers / "a.pdf").write_bytes(b"%PDF-1.4")

    monkeypatch.setattr(ee, "_extract_text_with_fallback", lambda _p, two_column_mode="auto", **kwargs: ("tiny", "pypdf", None, False))
    monkeypatch.setattr(
        ee, "_extract_text_tesseract_cli", lambda _p, timeout_sec=30, max_pages=20, lang="eng", profile="document": "good content " * 40
    )

    stats = ee.extract_from_papers_dir(
        str(papers),
        str(out),
        min_text_chars=20,
        ocr_enabled=True,
        ocr_min_chars_trigger=120,
    )
    assert stats["ocr_attempted_count"] == 1
    assert stats["ocr_succeeded_count"] == 1
    assert stats["ocr_failed_count"] == 0
    assert stats["created"] == 1


def test_ocr_rejected_on_low_confidence(tmp_path, monkeypatch) -> None:
    papers = tmp_path / "papers"
    out = tmp_path / "extracted"
    papers.mkdir()
    (papers / "a.pdf").write_bytes(b"%PDF-1.4")

    monkeypatch.setattr(ee, "_extract_text_with_fallback", lambda _p, two_column_mode="auto", **kwargs: ("tiny", "pypdf", None, False))
    monkeypatch.setattr(
        ee,
        "_extract_text_tesseract_cli",
        lambda _p, timeout_sec=30, max_pages=20, lang="eng", profile="document": ("good content " * 40, [], 20.0),
    )

    stats = ee.extract_from_papers_dir(
        str(papers),
        str(out),
        min_text_chars=1,
        ocr_enabled=True,
        ocr_min_chars_trigger=120,
        ocr_min_confidence=45.0,
    )
    assert stats["ocr_attempted_count"] == 1
    assert stats["ocr_failed_count"] == 1
    assert stats["ocr_rejected_low_confidence_count"] == 1
    assert stats["failed_reason_counts"]["ocr_low_confidence"] == 1
    assert stats["created"] == 1


def test_ocr_rejected_when_output_not_materially_better(tmp_path, monkeypatch) -> None:
    papers = tmp_path / "papers"
    out = tmp_path / "extracted"
    papers.mkdir()
    (papers / "a.pdf").write_bytes(b"%PDF-1.4")

    monkeypatch.setattr(ee, "_extract_text_with_fallback", lambda _p, two_column_mode="auto", **kwargs: ("tiny", "pypdf", None, False))
    monkeypatch.setattr(
        ee, "_extract_text_tesseract_cli", lambda _p, timeout_sec=30, max_pages=20, lang="eng", profile="document": "tiny better"
    )

    stats = ee.extract_from_papers_dir(
        str(papers),
        str(out),
        min_text_chars=4,
        ocr_enabled=True,
        ocr_min_chars_trigger=120,
        ocr_min_output_chars=200,
        ocr_min_gain_chars=40,
    )
    assert stats["ocr_attempted_count"] == 1
    assert stats["ocr_failed_count"] == 1
    assert stats["ocr_rejected_low_quality_count"] == 1
    assert stats["failed_reason_counts"]["ocr_low_quality"] == 1
    # Native path still used; extraction continues.
    assert stats["created"] == 1


def test_ocr_missing_binary_reason(tmp_path, monkeypatch) -> None:
    papers = tmp_path / "papers"
    out = tmp_path / "extracted"
    papers.mkdir()
    (papers / "a.pdf").write_bytes(b"%PDF-1.4")

    monkeypatch.setattr(ee, "_extract_text_with_fallback", lambda _p, two_column_mode="auto", **kwargs: ("tiny", "pypdf", None, False))
    monkeypatch.setattr(
        ee,
        "_extract_text_tesseract_cli",
        lambda _p, timeout_sec=30, max_pages=20, lang="eng", profile="document": (_ for _ in ()).throw(RuntimeError("ocr_binary_missing")),
    )

    stats = ee.extract_from_papers_dir(
        str(papers),
        str(out),
        min_text_chars=20,
        ocr_enabled=True,
        ocr_min_chars_trigger=120,
    )
    assert stats["ocr_attempted_count"] == 1
    assert stats["ocr_failed_count"] == 1
    assert stats["failed_reason_counts"]["ocr_binary_missing"] == 1


def test_ocr_timeout_reason(tmp_path, monkeypatch) -> None:
    papers = tmp_path / "papers"
    out = tmp_path / "extracted"
    papers.mkdir()
    (papers / "a.pdf").write_bytes(b"%PDF-1.4")

    monkeypatch.setattr(ee, "_extract_text_with_fallback", lambda _p, two_column_mode="auto", **kwargs: ("tiny", "pypdf", None, False))
    monkeypatch.setattr(
        ee,
        "_extract_text_tesseract_cli",
        lambda _p, timeout_sec=30, max_pages=20, lang="eng", profile="document": (_ for _ in ()).throw(RuntimeError("ocr_timeout")),
    )

    stats = ee.extract_from_papers_dir(
        str(papers),
        str(out),
        min_text_chars=20,
        ocr_enabled=True,
        ocr_min_chars_trigger=120,
    )
    assert stats["ocr_attempted_count"] == 1
    assert stats["ocr_failed_count"] == 1
    assert stats["failed_reason_counts"]["ocr_timeout"] == 1
