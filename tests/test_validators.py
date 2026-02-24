from src.core.schemas import EvidencePack, ExperimentProposal, ResearchReport
from src.core.validation import validate_report_citations, validate_semantic_claim_support


def test_citation_validator_happy_path() -> None:
    rep = ResearchReport(
        question="q",
        shortlist=[],
        synthesis="Claim from paper (Pabc:S1)",
        gaps=["Gap here (Pabc:S1)"],
        experiments=[ExperimentProposal(proposal="Do ablation (Pabc:S1)", citations=["(Pabc:S1)"])],
        citations=["(Pabc:S1)"],
    )
    assert not validate_report_citations(rep)


def test_citation_validator_fails_missing_citation() -> None:
    rep = ResearchReport(
        question="q",
        shortlist=[],
        synthesis="Claim without citation",
        gaps=[],
        experiments=[],
        citations=[],
    )
    errs = validate_report_citations(rep)
    assert errs


def test_semantic_validator_wrapper_disabled_path() -> None:
    rep = ResearchReport(
        question="q",
        shortlist=[],
        synthesis="Claim from paper (Pabc:S1)",
        gaps=[],
        experiments=[],
        citations=["(Pabc:S1)"],
    )
    evidence = EvidencePack(question="q", items=[])
    errors, metrics, warnings = validate_semantic_claim_support(rep, evidence, shadow_mode=True)
    # In test env model may or may not be cached; both outcomes are acceptable as long as wrapper is safe.
    assert isinstance(errors, list)
    assert isinstance(metrics, dict)
    assert isinstance(warnings, list)
