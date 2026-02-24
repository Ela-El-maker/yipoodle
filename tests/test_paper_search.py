from src.apps import paper_search
from src.apps.paper_search import _reconstruct_openalex_abstract


def test_reconstruct_openalex_abstract() -> None:
    inv = {
        "lightweight": [0],
        "segmentation": [1],
        "for": [2],
        "mobile": [3],
    }
    text = _reconstruct_openalex_abstract(inv)
    assert text == "lightweight segmentation for mobile"


class _JsonResp:
    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload


def test_search_crossref_parses_items(monkeypatch) -> None:
    monkeypatch.setattr(
        paper_search,
        "_request_get_with_retry",
        lambda *_args, **_kwargs: _JsonResp(
            {
                "message": {
                    "items": [
                        {
                            "title": ["Boundary-aware segmentation"],
                            "DOI": "10.1000/x",
                            "issued": {"date-parts": [[2024, 1, 1]]},
                            "author": [{"given": "A", "family": "B"}],
                            "container-title": ["CVPR"],
                            "URL": "https://doi.org/10.1000/x",
                            "is-referenced-by-count": 12,
                        }
                    ]
                }
            }
        ),
    )
    out = paper_search.search_crossref("segmentation", max_results=3)
    assert len(out) == 1
    assert out[0].source == "crossref"
    assert out[0].doi == "10.1000/x"
    assert out[0].citation_count == 12


def test_search_dblp_parses_hits(monkeypatch) -> None:
    monkeypatch.setattr(
        paper_search,
        "_request_get_with_retry",
        lambda *_args, **_kwargs: _JsonResp(
            {
                "result": {
                    "hits": {
                        "hit": [
                            {
                                "info": {
                                    "title": "Realtime matting",
                                    "year": "2023",
                                    "venue": "ECCV",
                                    "doi": "10.1000/y",
                                    "url": "https://dblp.org/rec/conf/eccv/abc",
                                    "authors": {"author": ["A One", "B Two"]},
                                }
                            }
                        ]
                    }
                }
            }
        ),
    )
    out = paper_search.search_dblp("matting", max_results=2)
    assert len(out) == 1
    assert out[0].source == "dblp"
    assert out[0].paper_id == "doi:10.1000/y"
    assert out[0].year == 2023


def test_search_core_parses_results(monkeypatch) -> None:
    monkeypatch.setattr(
        paper_search,
        "_request_get_with_retry",
        lambda *_args, **_kwargs: _JsonResp(
            {
                "results": [
                    {
                        "title": "Core CV paper",
                        "doi": "10.1000/core",
                        "yearPublished": 2022,
                        "authors": [{"name": "Core Author"}],
                        "downloadUrl": "https://example.org/core.pdf",
                        "abstract": "A paper from CORE",
                        "citationCount": 8,
                    }
                ]
            }
        ),
    )
    out = paper_search.search_core("computer vision", max_results=2)
    assert len(out) == 1
    assert out[0].source == "core"
    assert out[0].doi == "10.1000/core"
    assert str(out[0].url).endswith("core.pdf")


def test_search_openreview_parses_notes(monkeypatch) -> None:
    monkeypatch.setattr(
        paper_search,
        "_request_get_with_retry",
        lambda *_args, **_kwargs: _JsonResp(
            {
                "notes": [
                    {
                        "id": "abc123",
                        "content": {
                            "title": {"value": "OpenReview CV Note"},
                            "abstract": {"value": "Study on segmentation robustness"},
                            "authors": {"value": ["A", "B"]},
                            "venue": {"value": "ICLR 2026"},
                            "year": {"value": 2026},
                            "arxiv_id": {"value": "2601.12345"},
                        },
                    }
                ]
            }
        ),
    )
    out = paper_search.search_openreview("segmentation", max_results=2)
    assert len(out) == 1
    assert out[0].source == "openreview"
    assert out[0].title == "OpenReview CV Note"
    assert out[0].year == 2026
    assert out[0].arxiv_id == "2601.12345"


def test_search_github_parses_repositories(monkeypatch) -> None:
    monkeypatch.setattr(
        paper_search,
        "_request_get_with_retry",
        lambda *_args, **_kwargs: _JsonResp(
            {
                "items": [
                    {
                        "full_name": "org/repo",
                        "name": "repo",
                        "description": "Code for CV baseline",
                        "html_url": "https://github.com/org/repo",
                        "created_at": "2024-01-05T12:00:00Z",
                        "stargazers_count": 42,
                    }
                ]
            }
        ),
    )
    out = paper_search.search_github("cv model", max_results=2)
    assert len(out) == 1
    assert out[0].source == "github"
    assert out[0].citation_count == 42
    assert str(out[0].url).startswith("https://github.com/")


def test_search_zenodo_parses_hits(monkeypatch) -> None:
    monkeypatch.setattr(
        paper_search,
        "_request_get_with_retry",
        lambda *_args, **_kwargs: _JsonResp(
            {
                "hits": {
                    "hits": [
                        {
                            "id": 123,
                            "metadata": {
                                "title": "Zenodo CV Dataset",
                                "doi": "10.5281/zenodo.123",
                                "publication_date": "2023-06-01",
                                "creators": [{"name": "A. Researcher"}],
                                "description": "Dataset artifact",
                            },
                            "links": {"html": "https://zenodo.org/records/123"},
                            "files": [{"links": {"self": "https://zenodo.org/records/123/files/file.pdf"}}],
                        }
                    ]
                }
            }
        ),
    )
    out = paper_search.search_zenodo("segmentation", max_results=2)
    assert len(out) == 1
    assert out[0].source == "zenodo"
    assert out[0].doi == "10.5281/zenodo.123"
    assert str(out[0].url).endswith(".pdf")


def test_search_springer_parses_records(monkeypatch) -> None:
    monkeypatch.setattr(
        paper_search,
        "_request_get_with_retry",
        lambda *_args, **_kwargs: _JsonResp(
            {
                "records": [
                    {
                        "title": "Springer Vision Paper",
                        "doi": "10.1007/s001",
                        "publicationDate": "2022-03-11",
                        "creators": "A Author; B Author",
                        "publicationName": "Machine Vision and Applications",
                        "url": "https://link.springer.com/article/10.1007/s001",
                        "abstract": "Study abstract",
                    }
                ]
            }
        ),
    )
    out = paper_search.search_springer("vision", max_results=2)
    assert len(out) == 1
    assert out[0].source == "springer"
    assert out[0].year == 2022
    assert out[0].doi == "10.1007/s001"


def test_search_ieee_xplore_parses_articles(monkeypatch) -> None:
    monkeypatch.setattr(
        paper_search,
        "_request_get_with_retry",
        lambda *_args, **_kwargs: _JsonResp(
            {
                "articles": [
                    {
                        "title": "IEEE Vision Article",
                        "doi": "10.1109/12345",
                        "publication_year": "2021",
                        "publication_title": "IEEE Transactions",
                        "authors": {"authors": [{"full_name": "A Author"}]},
                        "pdf_url": "https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=1",
                    }
                ]
            }
        ),
    )
    out = paper_search.search_ieee_xplore("vision", max_results=2)
    assert len(out) == 1
    assert out[0].source == "ieee_xplore"
    assert out[0].year == 2021
    assert out[0].doi == "10.1109/12345"


def test_search_gdelt_parses_articles(monkeypatch) -> None:
    monkeypatch.setattr(
        paper_search,
        "_request_get_with_retry",
        lambda *_args, **_kwargs: _JsonResp(
            {
                "articles": [
                    {
                        "title": "Computer vision startup raises funding",
                        "url": "https://example.com/news",
                        "seendate": "20260223T120000Z",
                        "domain": "example.com",
                    }
                ]
            }
        ),
    )
    out = paper_search.search_gdelt("computer vision", max_results=2)
    assert len(out) == 1
    assert out[0].source == "gdelt"
    assert out[0].year == 2026
