from datetime import datetime, timezone
import json

from src.apps import paper_sync
from src.core.schemas import PaperRecord


def _paper(paper_id: str, source: str, url: str) -> PaperRecord:
    return PaperRecord(
        paper_id=paper_id,
        title=paper_id,
        authors=[],
        year=2025,
        venue=None,
        source=source,
        abstract="",
        pdf_path=None,
        url=url,
        doi=None,
        arxiv_id=paper_id.split(":")[-1] if source == "arxiv" else None,
        openalex_id=None,
        citation_count=0,
        is_open_access=False,
        sync_timestamp=datetime.now(timezone.utc),
    )


def test_sync_papers_download_categories(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(paper_sync, "init_db", lambda _db: None)
    monkeypatch.setattr(
        paper_sync,
        "search_arxiv",
        lambda *_args, **_kwargs: [_paper("arxiv:1", "arxiv", "https://arxiv.org/pdf/1.pdf")],
    )
    monkeypatch.setattr(
        paper_sync,
        "search_openalex",
        lambda *_args, **_kwargs: [_paper("oa:1", "openalex", "https://example.org/landing")],
    )
    monkeypatch.setattr(
        paper_sync,
        "search_semanticscholar",
        lambda *_args, **_kwargs: [_paper("s2:1", "semanticscholar", "https://example.org/maybe.pdf")],
    )
    monkeypatch.setattr(paper_sync, "upsert_papers", lambda _db, papers: len(papers))

    outcomes = {
        "https://arxiv.org/pdf/1.pdf": "downloaded",
        "https://example.org/landing": "non_pdf_content_type",
        "https://example.org/maybe.pdf": "download_http_error",
    }

    def fake_download(url: str, _dest: str) -> str:
        return outcomes[url]

    monkeypatch.setattr(paper_sync, "download_pdf_with_status", fake_download)

    stats = paper_sync.sync_papers(
        query="q",
        max_results=5,
        db_path=str(tmp_path / "papers.db"),
        papers_dir=str(tmp_path / "papers"),
        with_semantic_scholar=True,
    )
    assert stats["fetched"] == 3
    assert stats["downloaded"] == 1
    assert stats["non_pdf_content_type"] == 1
    assert stats["download_http_error"] == 1


def test_sync_papers_require_pdf_filter(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(paper_sync, "init_db", lambda _db: None)
    monkeypatch.setattr(
        paper_sync,
        "search_arxiv",
        lambda *_args, **_kwargs: [_paper("arxiv:1", "arxiv", "https://arxiv.org/abs/1")],
    )
    monkeypatch.setattr(
        paper_sync,
        "search_openalex",
        lambda *_args, **_kwargs: [_paper("oa:1", "openalex", "https://example.org/landing")],
    )
    monkeypatch.setattr(paper_sync, "search_semanticscholar", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(paper_sync, "upsert_papers", lambda _db, papers: len(papers))
    monkeypatch.setattr(paper_sync, "download_pdf_with_status", lambda *_args, **_kwargs: "downloaded")

    stats = paper_sync.sync_papers(
        query="q",
        max_results=5,
        db_path=str(tmp_path / "papers.db"),
        papers_dir=str(tmp_path / "papers"),
        require_pdf=True,
    )
    assert stats["fetched"] == 1
    assert stats["require_pdf_filtered"] == 1


def test_sync_papers_honors_sources_yaml_limits_and_enable(monkeypatch, tmp_path) -> None:
    cfg_path = tmp_path / "sources.yaml"
    cfg_path.write_text(
        json.dumps(
            {
                "sources": {
                    "arxiv": {"enabled": True, "max_results": 1},
                    "openalex": {"enabled": False, "max_results": 5},
                    "semanticscholar": {"enabled": True, "max_results": 5},
                    "paperswithcode": {"enabled": False},
                    "core": {"enabled": False},
                    "openreview": {"enabled": False},
                    "foo_source": {"enabled": True},
                },
                "limits": {"max_total_results": 2, "max_pdf_downloads": 1},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(paper_sync, "init_db", lambda _db: None)
    monkeypatch.setattr(
        paper_sync,
        "search_arxiv",
        lambda *_args, **kwargs: [
            _paper("arxiv:1", "arxiv", "https://arxiv.org/pdf/1.pdf"),
            _paper("arxiv:2", "arxiv", "https://arxiv.org/pdf/2.pdf"),
        ][: int(kwargs.get("max_results", 20))],
    )
    monkeypatch.setattr(
        paper_sync,
        "search_openalex",
        lambda *_args, **_kwargs: [_paper("oa:1", "openalex", "https://example.org/oa.pdf")],
    )
    monkeypatch.setattr(
        paper_sync,
        "search_semanticscholar",
        lambda *_args, **_kwargs: [_paper("s2:1", "semanticscholar", "https://example.org/s2.pdf")],
    )
    monkeypatch.setattr(
        paper_sync,
        "search_paperswithcode",
        lambda *_args, **_kwargs: [],
    )
    monkeypatch.setattr(
        paper_sync,
        "search_core",
        lambda *_args, **_kwargs: [],
    )
    monkeypatch.setattr(
        paper_sync,
        "search_openreview",
        lambda *_args, **_kwargs: [],
    )
    monkeypatch.setattr(paper_sync, "upsert_papers", lambda _db, papers: len(papers))
    calls = {"downloaded": 0}

    def fake_download(_url: str, _dest: str) -> str:
        calls["downloaded"] += 1
        return "downloaded"

    monkeypatch.setattr(paper_sync, "download_pdf_with_status", fake_download)

    stats = paper_sync.sync_papers(
        query="q",
        max_results=5,
        db_path=str(tmp_path / "papers.db"),
        papers_dir=str(tmp_path / "papers"),
        with_semantic_scholar=True,
        sources_config_path=str(cfg_path),
    )
    assert stats["from_arxiv"] == 1
    assert stats["from_openalex"] == 0
    assert stats["from_semanticscholar"] == 1
    assert stats["fetched"] == 2
    assert stats["downloaded"] == 1
    assert calls["downloaded"] == 1
    assert stats["sources_config_applied"] == 1
    assert stats["unsupported_enabled_sources_count"] == 1
    assert stats["unsupported_enabled_sources"] == ["foo_source"]


def test_sync_papers_unpaywall_enriches_doi_pdf(monkeypatch, tmp_path) -> None:
    cfg_path = tmp_path / "sources.yaml"
    cfg_path.write_text(
        json.dumps(
            {
                "sources": {
                    "arxiv": {"enabled": False},
                    "openalex": {"enabled": False},
                    "semanticscholar": {"enabled": False},
                    "crossref": {"enabled": True},
                    "dblp": {"enabled": False},
                    "paperswithcode": {"enabled": False},
                    "core": {"enabled": False},
                    "openreview": {"enabled": False},
                    "unpaywall": {"enabled": True, "required_query_params": {"email": "a@b.com"}},
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(paper_sync, "init_db", lambda _db: None)
    monkeypatch.setattr(
        paper_sync,
        "search_arxiv",
        lambda *_args, **_kwargs: [],
    )
    monkeypatch.setattr(
        paper_sync,
        "search_openalex",
        lambda *_args, **_kwargs: [],
    )
    monkeypatch.setattr(
        paper_sync,
        "search_semanticscholar",
        lambda *_args, **_kwargs: [],
    )
    monkeypatch.setattr(
        paper_sync,
        "search_crossref",
        lambda *_args, **_kwargs: [
            _paper("doi:10.1000/x", "crossref", "https://doi.org/10.1000/x").model_copy(update={"doi": "10.1000/x"})
        ],
    )
    monkeypatch.setattr(
        paper_sync,
        "search_dblp",
        lambda *_args, **_kwargs: [],
    )
    monkeypatch.setattr(
        paper_sync,
        "search_paperswithcode",
        lambda *_args, **_kwargs: [],
    )
    monkeypatch.setattr(
        paper_sync,
        "search_core",
        lambda *_args, **_kwargs: [],
    )
    monkeypatch.setattr(
        paper_sync,
        "search_openreview",
        lambda *_args, **_kwargs: [],
    )
    captured = {"url": None}

    def fake_upsert(_db: str, papers: list[PaperRecord]) -> int:
        captured["url"] = str(papers[0].url) if papers else None
        return len(papers)

    monkeypatch.setattr(paper_sync, "upsert_papers", fake_upsert)
    monkeypatch.setattr(
        paper_sync,
        "_unpaywall_pdf_url",
        lambda doi, endpoint, email: ("https://example.org/paper.pdf", True),
    )
    monkeypatch.setattr(paper_sync, "download_pdf_with_status", lambda *_args, **_kwargs: "downloaded")

    stats = paper_sync.sync_papers(
        query="q",
        max_results=5,
        db_path=str(tmp_path / "papers.db"),
        papers_dir=str(tmp_path / "papers"),
        with_semantic_scholar=True,
        sources_config_path=str(cfg_path),
    )
    assert stats["unpaywall_enabled"] == 1
    assert stats["unpaywall_lookups"] == 1
    assert stats["unpaywall_enriched"] == 1
    assert captured["url"] == "https://example.org/paper.pdf"


def test_sync_papers_paperswithcode_source_enabled(monkeypatch, tmp_path) -> None:
    cfg_path = tmp_path / "sources.yaml"
    cfg_path.write_text(
        json.dumps(
            {
                "sources": {
                    "arxiv": {"enabled": False},
                    "openalex": {"enabled": False},
                    "semanticscholar": {"enabled": False},
                    "crossref": {"enabled": False},
                    "dblp": {"enabled": False},
                    "paperswithcode": {"enabled": True, "max_results": 2},
                    "core": {"enabled": False},
                    "openreview": {"enabled": False},
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(paper_sync, "init_db", lambda _db: None)
    monkeypatch.setattr(paper_sync, "search_arxiv", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(paper_sync, "search_openalex", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(paper_sync, "search_semanticscholar", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(paper_sync, "search_crossref", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(paper_sync, "search_dblp", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(paper_sync, "search_core", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(paper_sync, "search_openreview", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(
        paper_sync,
        "search_paperswithcode",
        lambda *_args, **kwargs: [
            _paper("pwc:1", "paperswithcode", "https://example.org/1.pdf"),
            _paper("pwc:2", "paperswithcode", "https://example.org/2.pdf"),
            _paper("pwc:3", "paperswithcode", "https://example.org/3.pdf"),
        ][: int(kwargs.get("max_results", 20))],
    )
    monkeypatch.setattr(paper_sync, "upsert_papers", lambda _db, papers: len(papers))
    monkeypatch.setattr(paper_sync, "download_pdf_with_status", lambda *_args, **_kwargs: "downloaded")

    stats = paper_sync.sync_papers(
        query="q",
        max_results=5,
        db_path=str(tmp_path / "papers.db"),
        papers_dir=str(tmp_path / "papers"),
        with_semantic_scholar=True,
        sources_config_path=str(cfg_path),
    )
    assert stats["from_paperswithcode"] == 2
    assert stats["fetched"] == 2


def test_sync_papers_core_source_enabled(monkeypatch, tmp_path) -> None:
    cfg_path = tmp_path / "sources.yaml"
    cfg_path.write_text(
        json.dumps(
            {
                "sources": {
                    "arxiv": {"enabled": False},
                    "openalex": {"enabled": False},
                    "semanticscholar": {"enabled": False},
                    "crossref": {"enabled": False},
                    "dblp": {"enabled": False},
                    "paperswithcode": {"enabled": False},
                    "core": {"enabled": True, "max_results": 2, "endpoint": "https://api.core.ac.uk/v3/search/works"},
                    "openreview": {"enabled": False},
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(paper_sync, "init_db", lambda _db: None)
    monkeypatch.setattr(paper_sync, "search_arxiv", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(paper_sync, "search_openalex", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(paper_sync, "search_semanticscholar", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(paper_sync, "search_crossref", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(paper_sync, "search_dblp", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(paper_sync, "search_paperswithcode", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(paper_sync, "search_openreview", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(
        paper_sync,
        "search_core",
        lambda *_args, **kwargs: [
            _paper("core:1", "core", "https://example.org/c1.pdf"),
            _paper("core:2", "core", "https://example.org/c2.pdf"),
            _paper("core:3", "core", "https://example.org/c3.pdf"),
        ][: int(kwargs.get("max_results", 20))],
    )
    monkeypatch.setattr(paper_sync, "upsert_papers", lambda _db, papers: len(papers))
    monkeypatch.setattr(paper_sync, "download_pdf_with_status", lambda *_args, **_kwargs: "downloaded")

    stats = paper_sync.sync_papers(
        query="q",
        max_results=5,
        db_path=str(tmp_path / "papers.db"),
        papers_dir=str(tmp_path / "papers"),
        with_semantic_scholar=True,
        sources_config_path=str(cfg_path),
    )
    assert stats["from_core"] == 2
    assert stats["fetched"] == 2


def test_sync_papers_openreview_source_enabled(monkeypatch, tmp_path) -> None:
    cfg_path = tmp_path / "sources.yaml"
    cfg_path.write_text(
        json.dumps(
            {
                "sources": {
                    "arxiv": {"enabled": False},
                    "openalex": {"enabled": False},
                    "semanticscholar": {"enabled": False},
                    "crossref": {"enabled": False},
                    "dblp": {"enabled": False},
                    "paperswithcode": {"enabled": False},
                    "core": {"enabled": False},
                    "openreview": {"enabled": True, "max_results": 2},
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(paper_sync, "init_db", lambda _db: None)
    monkeypatch.setattr(paper_sync, "search_arxiv", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(paper_sync, "search_openalex", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(paper_sync, "search_semanticscholar", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(paper_sync, "search_crossref", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(paper_sync, "search_dblp", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(paper_sync, "search_paperswithcode", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(paper_sync, "search_core", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(
        paper_sync,
        "search_openreview",
        lambda *_args, **kwargs: [
            _paper("or:1", "openreview", "https://openreview.net/forum?id=1"),
            _paper("or:2", "openreview", "https://openreview.net/forum?id=2"),
            _paper("or:3", "openreview", "https://openreview.net/forum?id=3"),
        ][: int(kwargs.get("max_results", 20))],
    )
    monkeypatch.setattr(paper_sync, "upsert_papers", lambda _db, papers: len(papers))
    monkeypatch.setattr(paper_sync, "download_pdf_with_status", lambda *_args, **_kwargs: "downloaded")

    stats = paper_sync.sync_papers(
        query="q",
        max_results=5,
        db_path=str(tmp_path / "papers.db"),
        papers_dir=str(tmp_path / "papers"),
        with_semantic_scholar=True,
        sources_config_path=str(cfg_path),
    )
    assert stats["from_openreview"] == 2
    assert stats["fetched"] == 2


def test_sync_papers_github_and_zenodo_sources_enabled(monkeypatch, tmp_path) -> None:
    cfg_path = tmp_path / "sources.yaml"
    cfg_path.write_text(
        json.dumps(
            {
                "sources": {
                    "arxiv": {"enabled": False},
                    "openalex": {"enabled": False},
                    "semanticscholar": {"enabled": False},
                    "crossref": {"enabled": False},
                    "dblp": {"enabled": False},
                    "paperswithcode": {"enabled": False},
                    "core": {"enabled": False},
                    "openreview": {"enabled": False},
                    "github": {"enabled": True, "max_results": 2},
                    "zenodo": {"enabled": True, "max_results": 1},
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(paper_sync, "init_db", lambda _db: None)
    monkeypatch.setattr(paper_sync, "search_arxiv", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(paper_sync, "search_openalex", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(paper_sync, "search_semanticscholar", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(paper_sync, "search_crossref", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(paper_sync, "search_dblp", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(paper_sync, "search_paperswithcode", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(paper_sync, "search_core", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(paper_sync, "search_openreview", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(paper_sync, "search_opencitations", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(paper_sync, "search_springer", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(paper_sync, "search_ieee_xplore", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(
        paper_sync,
        "search_github",
        lambda *_args, **kwargs: [
            _paper("gh:1", "github", "https://github.com/org/repo1"),
            _paper("gh:2", "github", "https://github.com/org/repo2"),
            _paper("gh:3", "github", "https://github.com/org/repo3"),
        ][: int(kwargs.get("max_results", 20))],
    )
    monkeypatch.setattr(
        paper_sync,
        "search_zenodo",
        lambda *_args, **kwargs: [
            _paper("zen:1", "zenodo", "https://zenodo.org/records/1/files/paper.pdf"),
            _paper("zen:2", "zenodo", "https://zenodo.org/records/2/files/paper.pdf"),
        ][: int(kwargs.get("max_results", 20))],
    )
    monkeypatch.setattr(paper_sync, "upsert_papers", lambda _db, papers: len(papers))
    monkeypatch.setattr(paper_sync, "download_pdf_with_status", lambda *_args, **_kwargs: "downloaded")

    stats = paper_sync.sync_papers(
        query="q",
        max_results=5,
        db_path=str(tmp_path / "papers.db"),
        papers_dir=str(tmp_path / "papers"),
        with_semantic_scholar=True,
        sources_config_path=str(cfg_path),
    )
    assert stats["from_github"] == 2
    assert stats["from_zenodo"] == 1
    assert stats["fetched"] == 3


def test_sync_papers_springer_and_ieee_sources_enabled(monkeypatch, tmp_path) -> None:
    cfg_path = tmp_path / "sources.yaml"
    cfg_path.write_text(
        json.dumps(
            {
                "sources": {
                    "arxiv": {"enabled": False},
                    "openalex": {"enabled": False},
                    "semanticscholar": {"enabled": False},
                    "crossref": {"enabled": False},
                    "dblp": {"enabled": False},
                    "paperswithcode": {"enabled": False},
                    "core": {"enabled": False},
                    "openreview": {"enabled": False},
                    "springer": {"enabled": True, "max_results": 2},
                    "ieee_xplore": {"enabled": True, "max_results": 1},
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(paper_sync, "init_db", lambda _db: None)
    monkeypatch.setattr(paper_sync, "search_arxiv", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(paper_sync, "search_openalex", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(paper_sync, "search_semanticscholar", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(paper_sync, "search_crossref", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(paper_sync, "search_dblp", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(paper_sync, "search_paperswithcode", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(paper_sync, "search_core", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(paper_sync, "search_openreview", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(paper_sync, "search_github", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(paper_sync, "search_zenodo", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(paper_sync, "search_opencitations", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(paper_sync, "search_figshare", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(paper_sync, "search_openml", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(paper_sync, "search_gdelt", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(paper_sync, "search_wikidata", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(paper_sync, "search_orcid", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(
        paper_sync,
        "search_springer",
        lambda *_args, **kwargs: [
            _paper("sp:1", "springer", "https://link.springer.com/1"),
            _paper("sp:2", "springer", "https://link.springer.com/2"),
            _paper("sp:3", "springer", "https://link.springer.com/3"),
        ][: int(kwargs.get("max_results", 20))],
    )
    monkeypatch.setattr(
        paper_sync,
        "search_ieee_xplore",
        lambda *_args, **kwargs: [
            _paper("ie:1", "ieee_xplore", "https://ieeexplore.ieee.org/1"),
            _paper("ie:2", "ieee_xplore", "https://ieeexplore.ieee.org/2"),
        ][: int(kwargs.get("max_results", 20))],
    )
    monkeypatch.setattr(paper_sync, "upsert_papers", lambda _db, papers: len(papers))
    monkeypatch.setattr(paper_sync, "download_pdf_with_status", lambda *_args, **_kwargs: "downloaded")

    stats = paper_sync.sync_papers(
        query="q",
        max_results=5,
        db_path=str(tmp_path / "papers.db"),
        papers_dir=str(tmp_path / "papers"),
        with_semantic_scholar=True,
        sources_config_path=str(cfg_path),
    )
    assert stats["from_springer"] == 2
    assert stats["from_ieee_xplore"] == 1
    assert stats["fetched"] == 3


def test_sync_papers_normalizes_doi_and_uses_cache(monkeypatch, tmp_path) -> None:
    cfg_path = tmp_path / "sources.yaml"
    cache_path = tmp_path / "unpaywall_cache.json"
    cfg_path.write_text(
        json.dumps(
            {
                "sources": {
                    "arxiv": {"enabled": False},
                    "openalex": {"enabled": False},
                    "semanticscholar": {"enabled": False},
                    "crossref": {"enabled": True},
                    "unpaywall": {"enabled": True, "required_query_params": {"email": "a@b.com"}},
                },
                "cache": {"unpaywall_cache_path": str(cache_path)},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(paper_sync, "init_db", lambda _db: None)
    monkeypatch.setattr(paper_sync, "search_arxiv", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(paper_sync, "search_openalex", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(paper_sync, "search_semanticscholar", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(
        paper_sync,
        "search_crossref",
        lambda *_args, **_kwargs: [
            _paper("doi:10.1000/XYZ", "crossref", "https://doi.org/10.1000/XYZ").model_copy(update={"doi": "https://doi.org/10.1000/XYZ"})
        ],
    )
    monkeypatch.setattr(paper_sync, "upsert_papers", lambda _db, papers: len(papers))
    calls = {"n": 0}

    def _fake_unpaywall(doi: str, _endpoint: str, _email: str) -> tuple[str | None, bool]:
        calls["n"] += 1
        assert doi == "10.1000/xyz"
        return ("https://example.org/ok.pdf", True)

    monkeypatch.setattr(paper_sync, "_unpaywall_pdf_url", _fake_unpaywall)
    monkeypatch.setattr(paper_sync, "download_pdf_with_status", lambda *_args, **_kwargs: "downloaded")

    stats1 = paper_sync.sync_papers(
        query="q",
        max_results=3,
        db_path=str(tmp_path / "papers.db"),
        papers_dir=str(tmp_path / "papers"),
        sources_config_path=str(cfg_path),
    )
    stats2 = paper_sync.sync_papers(
        query="q",
        max_results=3,
        db_path=str(tmp_path / "papers.db"),
        papers_dir=str(tmp_path / "papers"),
        sources_config_path=str(cfg_path),
    )
    assert calls["n"] == 1
    assert stats1["unpaywall_cache_misses"] == 1
    assert stats2["unpaywall_cache_hits"] == 1
    assert cache_path.exists()


def test_sync_papers_source_quality_gate_metrics(monkeypatch, tmp_path) -> None:
    cfg_path = tmp_path / "sources.yaml"
    cfg_path.write_text(
        json.dumps(
            {
                "sources": {
                    "arxiv": {"enabled": True},
                    "openalex": {"enabled": False},
                    "semanticscholar": {"enabled": False},
                },
                "limits": {
                    "source_quality": {
                        "min_download_success_rate": 0.8,
                        "max_download_http_error_rate": 0.1,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(paper_sync, "init_db", lambda _db: None)
    monkeypatch.setattr(
        paper_sync,
        "search_arxiv",
        lambda *_args, **_kwargs: [
            _paper("arxiv:1", "arxiv", "https://arxiv.org/pdf/1.pdf"),
            _paper("arxiv:2", "arxiv", "https://arxiv.org/pdf/2.pdf"),
        ],
    )
    monkeypatch.setattr(paper_sync, "search_openalex", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(paper_sync, "search_semanticscholar", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(paper_sync, "upsert_papers", lambda _db, papers: len(papers))
    monkeypatch.setattr(
        paper_sync,
        "download_pdf_with_status",
        lambda url, _dest: "downloaded" if url.endswith("1.pdf") else "download_http_error",
    )

    stats = paper_sync.sync_papers(
        query="q",
        max_results=5,
        db_path=str(tmp_path / "papers.db"),
        papers_dir=str(tmp_path / "papers"),
        sources_config_path=str(cfg_path),
    )
    assert stats["source_quality_healthy"] is False
    reasons = stats["source_quality_reasons"]
    assert any("download_success_rate_below_threshold" in x for x in reasons)
    assert any("download_http_error_rate_above_threshold" in x for x in reasons)
