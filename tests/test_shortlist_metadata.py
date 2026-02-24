from src.apps.research_copilot import build_research_report
from src.core.schemas import EvidencePack


def test_shortlist_reason_includes_metadata_signals() -> None:
    evidence = EvidencePack(
        question="q",
        items=[
            {
                "paper_id": "p1",
                "snippet_id": "Pp1:S1",
                "score": 1.2,
                "section": "results",
                "text": "A cited claim 30 FPS.",
                "paper_year": 2024,
                "paper_venue": "CVPR",
                "citation_count": 320,
            },
            {
                "paper_id": "p1",
                "snippet_id": "Pp1:S2",
                "score": 0.8,
                "section": "limitations",
                "text": "Limitation found.",
                "paper_year": 2024,
                "paper_venue": "CVPR",
                "citation_count": 320,
            },
        ],
    )
    report = build_research_report("q", evidence, min_items=1, min_score=0.1)
    assert report.shortlist
    reason = report.shortlist[0].reason
    assert "year=2024" in reason
    assert "venue=CVPR" in reason
    assert "citations=320" in reason
