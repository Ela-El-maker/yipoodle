from src.core.ids import normalize_id, snippet_id


def test_snippet_id_format() -> None:
    sid = snippet_id("doi:10.1000/test", 2)
    assert sid == "Pdoi_10_1000_test:S2"
    assert normalize_id("arxiv:1234.5678") == "arxiv_1234_5678"
