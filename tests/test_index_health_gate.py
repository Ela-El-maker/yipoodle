import json

import pytest

from src.apps.index_builder import build_index


def test_build_index_requires_healthy_corpus_passes_when_thresholds_allow() -> None:
    stats = build_index(
        corpus_dir="tests/fixtures/extracted",
        out_path="runs/tmp_idx_health_pass.json",
        require_healthy_corpus=True,
        min_snippets=1,
        min_avg_chars_per_paper=1,
        min_avg_chars_per_page=0,
        max_extract_error_rate=1.0,
    )
    assert stats["snippets"] == 4
    assert "corpus_health" in stats
    assert stats["corpus_health"]["healthy"] is True


def test_build_index_requires_healthy_corpus_fails_on_unhealthy(tmp_path) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "p1.json").write_text(json.dumps({"paper": {"paper_id": "p1"}, "snippets": []}), encoding="utf-8")
    with pytest.raises(ValueError, match="Corpus health check failed before build-index"):
        build_index(
            corpus_dir=str(corpus),
            out_path=str(tmp_path / "idx.json"),
            require_healthy_corpus=True,
            min_snippets=1,
            min_avg_chars_per_paper=1,
            min_avg_chars_per_page=0,
            max_extract_error_rate=1.0,
        )
