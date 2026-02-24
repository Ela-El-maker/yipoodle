from src.core.schemas import EvidencePack, ResearchReport
from src.core.validation import validate_claim_support


def test_claim_support_passes_for_overlapping_claim() -> None:
    ev = EvidencePack(
        question="q",
        items=[
            {
                "paper_id": "p1",
                "snippet_id": "Pp1:S1",
                "score": 1.0,
                "section": "results",
                "text": "Model runs at 72 FPS on mobile GPU with strong edge quality.",
            }
        ],
    )
    rep = ResearchReport(
        question="q",
        shortlist=[],
        synthesis="Claim: model runs at 72 FPS on mobile GPU (Pp1:S1)",
        gaps=[],
        experiments=[],
        citations=["(Pp1:S1)"],
    )
    assert not validate_claim_support(rep, ev)


def test_claim_support_flags_low_overlap() -> None:
    ev = EvidencePack(
        question="q",
        items=[
            {
                "paper_id": "p1",
                "snippet_id": "Pp1:S1",
                "score": 1.0,
                "section": "results",
                "text": "Model runs at 72 FPS on mobile GPU.",
            }
        ],
    )
    rep = ResearchReport(
        question="q",
        shortlist=[],
        synthesis="Claim: transformer language model dominates legal reasoning benchmarks (Pp1:S1)",
        gaps=[],
        experiments=[],
        citations=["(Pp1:S1)"],
    )
    errs = validate_claim_support(rep, ev)
    assert errs
