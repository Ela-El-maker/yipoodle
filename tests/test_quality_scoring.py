from src.apps.evidence_extract import compute_extraction_quality


def test_quality_scoring_good_text() -> None:
    text = """
Abstract
This paper presents a practical method for mobile segmentation with robust boundary handling.
Introduction
We evaluate on public datasets and report latency and quality metrics.
References
""".strip()
    score, band, signals = compute_extraction_quality(text)
    assert score >= 0.45
    assert band in {"ok", "good"}
    assert signals["compact_len"] > 0


def test_quality_scoring_poor_text() -> None:
    text = "\x00\x01\x02 ???"
    score, band, _signals = compute_extraction_quality(text)
    assert score < 0.45
    assert band == "poor"
