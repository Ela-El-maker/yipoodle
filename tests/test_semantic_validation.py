from __future__ import annotations

import numpy as np

from src.core.schemas import EvidenceItem, EvidencePack, ResearchReport
from src.core.semantic_validation import compute_claim_evidence_semantic_scores, validate_semantic_claim_support
from src.core.validation import validate_semantic_claim_support as validate_semantic_wrapper


class _FakeModel:
    def encode(self, texts, convert_to_numpy=True, normalize_embeddings=True):  # noqa: ANN001
        del convert_to_numpy, normalize_embeddings
        claim = texts[0].lower()
        evidence = texts[1].lower()
        # Paraphrase-ish pair -> high similarity
        if "edge artifacts" in claim and "boundary artifacts" in evidence:
            return np.array([[1.0, 0.0], [0.9, 0.1]])
        # Contradictory-ish pair -> moderate similarity to test contradiction proxy
        if "not improve" in claim and "improves" in evidence:
            return np.array([[1.0, 0.0], [0.8, 0.2]])
        # default low similarity
        return np.array([[1.0, 0.0], [-1.0, 0.0]])


def _mk_report_and_evidence(line: str, evidence_text: str) -> tuple[ResearchReport, EvidencePack]:
    rep = ResearchReport(question="q", shortlist=[], synthesis=line, gaps=[], experiments=[], citations=["(P1:S1)"])
    ev = EvidencePack(question="q", items=[EvidenceItem(paper_id="P1", snippet_id="P1:S1", score=1.0, section="results", text=evidence_text)])
    return rep, ev


def test_compute_semantic_scores_paraphrase_high_support(monkeypatch) -> None:
    monkeypatch.setattr("src.core.semantic_validation._load_model", lambda _m: _FakeModel())
    out = compute_claim_evidence_semantic_scores(
        "Main edge artifacts increase around thin structures.",
        "Boundary artifacts are more common on thin structures.",
        model_name="sentence-transformers/all-MiniLM-L6-v2",
    )
    assert out["support_score"] > 0.8


def test_validate_semantic_claim_support_flags_contradiction(monkeypatch) -> None:
    monkeypatch.setattr("src.core.semantic_validation._load_model", lambda _m: _FakeModel())
    rep, ev = _mk_report_and_evidence(
        "Model does not improve accuracy on the benchmark (P1:S1)",
        "The model improves accuracy on the benchmark by 2 points.",
    )
    result = validate_semantic_claim_support(rep, ev, min_support=0.55, max_contradiction=0.30)
    assert result.lines_contradiction >= 1


def test_validate_semantic_wrapper_shadow_warn(monkeypatch) -> None:
    monkeypatch.setattr("src.core.semantic_validation._load_model", lambda _m: _FakeModel())
    rep, ev = _mk_report_and_evidence(
        "Model does not improve accuracy on the benchmark (P1:S1)",
        "The model improves accuracy on the benchmark by 2 points.",
    )
    errors, metrics, warnings = validate_semantic_wrapper(
        rep,
        ev,
        min_support=0.55,
        max_contradiction=0.30,
        shadow_mode=True,
        fail_on_low_support=False,
    )
    assert not errors
    assert metrics["semantic_checked"] is True
    assert metrics["semantic_status"] == "warn"
    assert warnings


def test_validate_semantic_wrapper_fail_mode(monkeypatch) -> None:
    monkeypatch.setattr("src.core.semantic_validation._load_model", lambda _m: _FakeModel())
    rep, ev = _mk_report_and_evidence(
        "Model does not improve accuracy on the benchmark (P1:S1)",
        "The model improves accuracy on the benchmark by 2 points.",
    )
    errors, metrics, warnings = validate_semantic_wrapper(
        rep,
        ev,
        min_support=0.55,
        max_contradiction=0.30,
        shadow_mode=False,
        fail_on_low_support=True,
    )
    assert errors
    assert metrics["semantic_status"] == "fail"
    assert not warnings


def test_validate_semantic_wrapper_online_mode(monkeypatch) -> None:
    monkeypatch.setattr("src.core.semantic_validation._load_model", lambda _m: _FakeModel())

    class _OnlineResult:
        checked_lines = 1
        support_avg = 0.82
        support_min = 0.82
        contradiction_max = 0.10
        lines_below_threshold = 0
        lines_contradiction = 0
        status = "pass"
        latency_ms = 12.3

    monkeypatch.setattr(
        "src.core.semantic_online.validate_online_semantic_claim_support",
        lambda *args, **kwargs: _OnlineResult(),
    )
    rep, ev = _mk_report_and_evidence(
        "Edge artifacts occur on thin regions (P1:S1)",
        "Boundary artifacts are common on thin structures.",
    )
    errors, metrics, warnings = validate_semantic_wrapper(
        rep,
        ev,
        semantic_mode="online",
        shadow_mode=True,
    )
    assert not errors
    assert not warnings
    assert metrics["online_semantic_checked"] is True
    assert metrics["semantic_mode"] == "online"
    assert metrics["semantic_status"] == "pass"


def test_validate_semantic_wrapper_online_failure_shadow(monkeypatch) -> None:
    monkeypatch.setattr("src.core.semantic_validation._load_model", lambda _m: _FakeModel())
    monkeypatch.setattr(
        "src.core.semantic_online.validate_online_semantic_claim_support",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    rep, ev = _mk_report_and_evidence(
        "Edge artifacts occur on thin regions (P1:S1)",
        "Boundary artifacts are common on thin structures.",
    )
    errors, metrics, warnings = validate_semantic_wrapper(
        rep,
        ev,
        semantic_mode="online",
        shadow_mode=True,
    )
    assert not errors
    assert warnings
    assert metrics["online_semantic_checked"] is False
