from __future__ import annotations

from dataclasses import dataclass

from src.apps.layout_engine import analyze_page_regions, reconstruct_page_text_v2


@dataclass
class _Rect:
    width: float
    height: float


class _Page:
    def __init__(self, blocks: list[tuple], width: float = 600.0, height: float = 1000.0) -> None:
        self._blocks = blocks
        self.rect = _Rect(width=width, height=height)

    def get_text(self, kind: str):
        assert kind == "blocks"
        return self._blocks


def test_layout_region_classification_detects_table_and_footnote() -> None:
    blocks = [
        (20.0, 40.0, 260.0, 70.0, "Left body paragraph one", 0, 0),
        (320.0, 50.0, 560.0, 80.0, "Right body paragraph one", 0, 0),
        (40.0, 400.0, 560.0, 440.0, "Year | Return | Volatility | Sharpe", 0, 0),
        (30.0, 900.0, 560.0, 930.0, "1) Footnote with DOI http://example.org", 0, 0),
    ]
    analysis = analyze_page_regions(_Page(blocks))
    assert analysis.region_counts["table"] >= 1
    assert analysis.region_counts["footnote"] >= 1
    assert analysis.region_counts["body_left"] >= 1
    assert analysis.region_counts["body_right"] >= 1


def test_layout_reconstruction_orders_body_then_table_then_footnote() -> None:
    blocks = [
        (20.0, 40.0, 260.0, 70.0, "Left body", 0, 0),
        (320.0, 50.0, 560.0, 80.0, "Right body", 0, 0),
        (40.0, 420.0, 560.0, 450.0, "A | B | C | 123", 0, 0),
        (30.0, 910.0, 560.0, 940.0, "1) note text", 0, 0),
    ]
    res = reconstruct_page_text_v2(analyze_page_regions(_Page(blocks)), table_handling="linearize", footnote_handling="append")
    text = res.text
    assert "Left body" in text
    assert "Right body" in text
    assert "A | B | C | 123" in text
    assert "1) note text" in text
    assert 0.0 <= res.confidence <= 1.0


def test_layout_confidence_penalizes_wide_crossing_body_blocks() -> None:
    blocks = [
        (20.0, 40.0, 560.0, 100.0, "This long body-like paragraph crosses both columns and may indicate mixed layout.", 0, 0),
        (20.0, 120.0, 560.0, 180.0, "Another crossing paragraph with mixed structure and embedded table cues 2024 10% 20% 30%.", 0, 0),
    ]
    res = reconstruct_page_text_v2(analyze_page_regions(_Page(blocks)), table_handling="linearize", footnote_handling="append")
    assert res.confidence < 0.55
