from src.apps import evidence_extract as ee


def test_extract_fallback_uses_second_extractor(monkeypatch) -> None:
    monkeypatch.setattr(
        ee,
        "_extract_text_pymupdf",
        lambda _p, two_column_mode="auto", layout_engine="shadow", layout_table_handling="linearize", layout_footnote_handling="append", layout_min_region_confidence=0.55: (_ for _ in ()).throw(RuntimeError("bad")),
    )
    monkeypatch.setattr(ee, "_extract_text_pdfminer", lambda _p: "ok from pdfminer")
    monkeypatch.setattr(ee, "_extract_text_pypdf", lambda _p: "ok from pypdf")

    text, extractor, reason, two_col, page_stats, layout_stats = ee._extract_text_with_fallback("/tmp/x.pdf")
    assert text == "ok from pdfminer"
    assert extractor == "pdfminer"
    assert reason is None
    assert two_col is False
    assert isinstance(page_stats, list)
    assert isinstance(layout_stats, dict)


def test_extract_fallback_missing_optional_deps(monkeypatch) -> None:
    monkeypatch.setattr(
        ee,
        "_extract_text_pymupdf",
        lambda _p, two_column_mode="auto", layout_engine="shadow", layout_table_handling="linearize", layout_footnote_handling="append", layout_min_region_confidence=0.55: (_ for _ in ()).throw(ModuleNotFoundError("no fitz")),
    )
    monkeypatch.setattr(ee, "_extract_text_pdfminer", lambda _p: (_ for _ in ()).throw(ModuleNotFoundError("no pdfminer")))
    monkeypatch.setattr(ee, "_extract_text_pypdf", lambda _p: (_ for _ in ()).throw(ModuleNotFoundError("no pypdf")))

    text, extractor, reason, _two_col, _page_stats, _layout_stats = ee._extract_text_with_fallback("/tmp/x.pdf")
    assert text is None
    assert extractor is None
    assert reason == "extractor_missing_dependency"


def test_extract_fallback_classifies_encrypted(monkeypatch) -> None:
    monkeypatch.setattr(
        ee,
        "_extract_text_pymupdf",
        lambda _p, two_column_mode="auto", layout_engine="shadow", layout_table_handling="linearize", layout_footnote_handling="append", layout_min_region_confidence=0.55: (_ for _ in ()).throw(RuntimeError("AES cryptography required")),
    )
    monkeypatch.setattr(ee, "_extract_text_pdfminer", lambda _p: (_ for _ in ()).throw(RuntimeError("parse fail")))
    monkeypatch.setattr(ee, "_extract_text_pypdf", lambda _p: (_ for _ in ()).throw(RuntimeError("parse fail")))

    text, extractor, reason, _two_col, _page_stats, _layout_stats = ee._extract_text_with_fallback("/tmp/x.pdf")
    assert text is None
    assert extractor is None
    assert reason == "encrypted_or_unsupported"
