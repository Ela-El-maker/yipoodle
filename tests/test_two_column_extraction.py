from src.apps import evidence_extract as ee


def test_detect_two_column_blocks_positive() -> None:
    blocks = []
    for i in range(6):
        blocks.append((10.0, float(i * 20), 240.0, float(i * 20 + 10), f"L{i}", 0, 0))
    for i in range(6):
        blocks.append((320.0, float(i * 20), 550.0, float(i * 20 + 10), f"R{i}", 0, 0))
    is_two_col, split = ee._detect_two_column_blocks(blocks, page_width=600.0)
    assert is_two_col is True
    assert 240.0 < split < 320.0


def test_detect_two_column_blocks_negative_single_column() -> None:
    blocks = []
    for i in range(10):
        blocks.append((30.0, float(i * 15), 500.0, float(i * 15 + 8), f"B{i}", 0, 0))
    is_two_col, split = ee._detect_two_column_blocks(blocks, page_width=600.0)
    assert is_two_col is False
    assert split == 0.0


def test_clean_blocks_filters_headers_footers_and_captions() -> None:
    blocks = [
        (10.0, 5.0, 580.0, 20.0, "Journal Header 2026", 0, 0),  # header
        (10.0, 100.0, 240.0, 130.0, "Valid left text", 0, 0),
        (320.0, 100.0, 560.0, 130.0, "Valid right text", 0, 0),
        (20.0, 200.0, 300.0, 230.0, "Figure 1. qualitative results", 0, 0),  # caption
        (20.0, 980.0, 560.0, 995.0, "Page 3", 0, 0),  # footer
    ]
    cleaned = ee._clean_blocks_for_layout(blocks, page_width=600.0, page_height=1000.0)
    texts = [b[4] for b in cleaned]
    assert "Valid left text" in texts
    assert "Valid right text" in texts
    assert all("Figure 1." not in t for t in texts)
    assert all("Journal Header" not in t for t in texts)
    assert all("Page 3" not in t for t in texts)
