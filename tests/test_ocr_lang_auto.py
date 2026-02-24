from src.apps import evidence_extract as ee


def test_detect_ocr_language_latin_english() -> None:
    txt = "This is a test document and the method is evaluated with data from multiple sources."
    lang, detected = ee._detect_ocr_language(txt)
    assert detected is True
    assert lang == "eng"


def test_detect_ocr_language_latin_spanish_mixed_with_english() -> None:
    txt = "La segmentacion de imagenes y el modelo en datos de prueba con analisis y resultados."
    lang, detected = ee._detect_ocr_language(txt)
    assert detected is True
    assert lang in {"eng+spa", "spa"}


def test_detect_ocr_language_short_text_fallback() -> None:
    lang, detected = ee._detect_ocr_language("abc")
    assert detected is False
    assert lang == "eng"


def test_extract_records_auto_lang_usage(tmp_path, monkeypatch) -> None:
    papers = tmp_path / "papers"
    out = tmp_path / "extracted"
    papers.mkdir()
    (papers / "a.pdf").write_bytes(b"%PDF-1.4")

    monkeypatch.setattr(
        ee,
        "_extract_text_with_fallback",
        lambda _p, two_column_mode="auto", **kwargs: ("la segmentacion en datos y resultados", "pypdf", None, False),
    )
    seen = {}

    def _fake_ocr(_p, timeout_sec=30, max_pages=20, lang="eng", profile="document"):
        seen["lang"] = lang
        return "good content " * 50

    monkeypatch.setattr(ee, "_extract_text_tesseract_cli", _fake_ocr)

    stats = ee.extract_from_papers_dir(
        str(papers),
        str(out),
        min_text_chars=20,
        ocr_enabled=True,
        ocr_min_chars_trigger=120,
        ocr_lang="auto",
    )
    assert stats["ocr_attempted_count"] == 1
    assert stats["ocr_lang_auto_detected_count"] in {0, 1}
    assert "lang" in seen
    assert seen["lang"] in {"eng", "eng+spa", "spa"}
