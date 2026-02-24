import json
from pathlib import Path

from src.apps.index_builder import build_index
from src.apps.paper_ingest import init_db, upsert_papers
from src.core.schemas import PaperRecord


def test_build_index_enriches_from_db(tmp_path) -> None:
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir(parents=True, exist_ok=True)

    paper_id = "doi:10.1000/testmeta"
    extracted = {
        "paper": {
            "paper_id": paper_id,
            "title": "T",
            "authors": [],
            "year": None,
            "venue": None,
            "source": "local",
            "abstract": "",
            "pdf_path": None,
            "url": "https://example.org",
            "doi": None,
            "arxiv_id": None,
            "openalex_id": None,
            "citation_count": 0,
            "is_open_access": False,
            "sync_timestamp": "2026-01-01T00:00:00Z"
        },
        "snippets": [
            {
                "snippet_id": "Pdoi_10_1000_testmeta:S1",
                "paper_id": paper_id,
                "section": "results",
                "text": "A result snippet",
                "page_hint": None,
                "token_count": 3,
                "paper_year": None,
                "paper_venue": None,
                "citation_count": 0,
            }
        ],
    }
    (corpus_dir / "sample.json").write_text(json.dumps(extracted), encoding="utf-8")

    db_path = tmp_path / "papers.db"
    init_db(str(db_path))
    p = PaperRecord(
        paper_id=paper_id,
        title="Paper",
        authors=["A"],
        year=2024,
        venue="CVPR",
        source="openalex",
        abstract="",
        pdf_path=None,
        url="https://example.org",
        doi="10.1000/testmeta",
        arxiv_id=None,
        openalex_id="openalex:abc",
        citation_count=77,
        is_open_access=True,
        sync_timestamp="2026-01-01T00:00:00Z",
    )
    upsert_papers(str(db_path), [p])

    out_index = tmp_path / "index.json"
    stats = build_index(str(corpus_dir), str(out_index), db_path=str(db_path))
    assert stats["enriched"] == 1

    payload = json.loads(out_index.read_text(encoding="utf-8"))
    sn = payload["snippets"][0]
    assert sn["paper_year"] == 2024
    assert sn["paper_venue"] == "CVPR"
    assert sn["citation_count"] == 77
