import json

from src.apps.index_builder import build_index


def test_build_index_appends_live_data(tmp_path) -> None:
    live_payload = {
        "snippets": [
            {
                "snippet_id": "SNAP:test:S1",
                "paper_id": "SNAP:test",
                "section": "live",
                "text": "live snippet text",
                "token_count": 3,
                "paper_year": None,
                "paper_venue": "demo_live",
                "citation_count": 0,
                "extraction_quality_score": 1.0,
                "extraction_quality_band": "good",
                "extraction_source": "native",
            }
        ]
    }
    live_path = tmp_path / "live.json"
    live_path.write_text(json.dumps(live_payload), encoding="utf-8")
    out_idx = tmp_path / "idx.json"

    stats = build_index(
        corpus_dir="tests/fixtures/extracted",
        out_path=str(out_idx),
        live_data_paths=[str(live_path)],
    )
    assert stats["live_snippets_added"] == 1
    payload = json.loads(out_idx.read_text(encoding="utf-8"))
    ids = [s["snippet_id"] for s in payload["snippets"]]
    assert "SNAP:test:S1" in ids
