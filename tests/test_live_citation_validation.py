from src.core.schemas import ExperimentProposal, ResearchReport
from src.core.validation import validate_report_citations


def test_validator_accepts_snapshot_citations() -> None:
    rep = ResearchReport(
        question="q",
        shortlist=[],
        synthesis="Claim from live source (SNAP:abc123:S1)",
        gaps=["Gap (SNAP:abc123:S1)"],
        experiments=[ExperimentProposal(proposal="Test (SNAP:abc123:S1)", citations=["(SNAP:abc123:S1)"])],
        citations=["(SNAP:abc123:S1)"],
    )
    assert not validate_report_citations(rep)
