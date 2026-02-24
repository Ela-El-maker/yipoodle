from src.core.schemas import EvidencePack, PaperRecord, ResearchReport, SnippetRecord


def test_schema_parse_sample_payloads() -> None:
    p = PaperRecord(
        paper_id="arxiv:1",
        title="t",
        authors=["a"],
        year=2024,
        venue="arXiv",
        source="arxiv",
        abstract="abs",
        pdf_path=None,
        url="https://arxiv.org/abs/1",
        doi=None,
        arxiv_id="1",
        sync_timestamp="2026-01-01T00:00:00Z",
    )
    s = SnippetRecord(
        snippet_id="Parxiv_1:S1",
        paper_id=p.paper_id,
        section="abstract",
        text="hello",
        page_hint=1,
        token_count=1,
    )
    ev = EvidencePack(question="q", items=[{"paper_id": p.paper_id, "snippet_id": s.snippet_id, "score": 1.0, "section": s.section, "text": s.text}])
    rr = ResearchReport(question="q", shortlist=[], synthesis="Not found in sources.", gaps=[], experiments=[], citations=[])

    assert p.paper_id
    assert ev.items[0].snippet_id == s.snippet_id
    assert rr.question == "q"
