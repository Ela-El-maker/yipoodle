from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import re
from typing import Any

import numpy as np

from src.core.schemas import EvidencePack, ResearchReport


_CIT_RE = re.compile(r"\(P[^:()\s]+:S\d+\)")
_TOK_RE = re.compile(r"[a-zA-Z0-9]+")
_NEG_WORDS = {"no", "not", "never", "without", "none", "cannot", "can't", "won't"}
_DIRECTION_PAIRS = [
    ("increase", "decrease"),
    ("improve", "degrade"),
    ("improves", "degrades"),
    ("higher", "lower"),
    ("more", "less"),
    ("gain", "loss"),
    ("better", "worse"),
]
_CONTRAST_MARKERS = {"however", "but", "whereas", "although", "despite"}


@dataclass
class SemanticLineResult:
    line_no: int
    citations: list[str]
    support_score: float
    contradiction_score: float
    status: str
    claim: str


@dataclass
class SemanticValidationResult:
    line_results: list[SemanticLineResult]
    support_avg: float | None
    support_min: float | None
    contradiction_max: float | None
    lines_below_threshold: int
    lines_contradiction: int
    status: str
    model_name: str
    warning: str | None = None


def _extract_citations(text: str) -> list[str]:
    return _CIT_RE.findall(text)


def _normalize_tokens(text: str) -> set[str]:
    return {t.lower() for t in _TOK_RE.findall(text)}


def _contains_negation(text: str) -> bool:
    toks = _normalize_tokens(text)
    return any(w in toks for w in _NEG_WORDS)


def _direction_conflict(claim: str, evidence: str) -> bool:
    c = _normalize_tokens(claim)
    e = _normalize_tokens(evidence)
    for a, b in _DIRECTION_PAIRS:
        if (a in c and b in e) or (b in c and a in e):
            return True
    return False


def _numeric_direction_conflict(claim: str, evidence: str) -> bool:
    c = claim.lower()
    e = evidence.lower()
    claim_dir_up = any(x in c for x in ("increased", "increase", "higher", "more"))
    claim_dir_down = any(x in c for x in ("decreased", "decrease", "lower", "less"))
    ev_dir_up = any(x in e for x in ("increased", "increase", "higher", "more"))
    ev_dir_down = any(x in e for x in ("decreased", "decrease", "lower", "less"))
    if claim_dir_up and ev_dir_down:
        return True
    if claim_dir_down and ev_dir_up:
        return True
    return False


def _contradiction_proxy(claim: str, evidence: str, support_score: float) -> float:
    score = 0.0
    c_neg = _contains_negation(claim)
    e_neg = _contains_negation(evidence)
    if c_neg != e_neg:
        score += 0.45
    if _direction_conflict(claim, evidence):
        score += 0.35
    if _numeric_direction_conflict(claim, evidence):
        score += 0.25
    if any(m in claim.lower() for m in _CONTRAST_MARKERS) and support_score < 0.5:
        score += 0.10
    score += 0.20 * (1.0 - support_score)
    return float(max(0.0, min(1.0, score)))


@lru_cache(maxsize=4)
def _load_model(model_name: str):
    try:
        from sentence_transformers import SentenceTransformer
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("sentence-transformers is required for semantic faithfulness checks") from exc
    return SentenceTransformer(model_name, local_files_only=True)


def compute_claim_evidence_semantic_scores(claim: str, evidence_text: str, model_name: str) -> dict[str, float]:
    if not claim.strip() or not evidence_text.strip():
        return {"support_score": 0.0, "contradiction_score": 0.0}
    model = _load_model(model_name)
    emb = model.encode([claim, evidence_text], convert_to_numpy=True, normalize_embeddings=True)
    sim = float(np.dot(emb[0], emb[1]))
    # Convert [-1,1] to [0,1] for easier thresholding.
    support = max(0.0, min(1.0, (sim + 1.0) / 2.0))
    contradiction = _contradiction_proxy(claim, evidence_text, support_score=support)
    return {"support_score": support, "contradiction_score": contradiction}


def validate_semantic_claim_support(
    report: ResearchReport,
    evidence: EvidencePack,
    *,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    min_support: float = 0.55,
    max_contradiction: float = 0.30,
) -> SemanticValidationResult:
    evidence_by_cit = {f"({item.snippet_id})": item.text for item in evidence.items}
    lines = [ln.strip() for ln in report.synthesis.splitlines() if ln.strip()]
    line_results: list[SemanticLineResult] = []
    supports: list[float] = []
    contradictions: list[float] = []

    for i, line in enumerate(lines, start=1):
        citations = _extract_citations(line)
        if not citations:
            continue
        claim = _CIT_RE.sub("", line).strip()
        if not claim:
            continue
        evidence_text = " ".join(evidence_by_cit.get(c, "") for c in citations).strip()
        if not evidence_text:
            continue
        scores = compute_claim_evidence_semantic_scores(claim, evidence_text, model_name=model_name)
        support = float(scores["support_score"])
        contradiction = float(scores["contradiction_score"])
        supports.append(support)
        contradictions.append(contradiction)
        if support < float(min_support):
            status = "weak_support"
        elif contradiction > float(max_contradiction):
            status = "possible_contradiction"
        else:
            status = "supported"
        line_results.append(
            SemanticLineResult(
                line_no=i,
                citations=citations,
                support_score=support,
                contradiction_score=contradiction,
                status=status,
                claim=claim,
            )
        )

    lines_below = sum(1 for r in line_results if r.status == "weak_support")
    lines_contra = sum(1 for r in line_results if r.status == "possible_contradiction")
    if lines_below == 0 and lines_contra == 0:
        overall = "pass"
    else:
        overall = "warn"
    return SemanticValidationResult(
        line_results=line_results,
        support_avg=(sum(supports) / len(supports)) if supports else None,
        support_min=min(supports) if supports else None,
        contradiction_max=max(contradictions) if contradictions else None,
        lines_below_threshold=lines_below,
        lines_contradiction=lines_contra,
        status=overall,
        model_name=model_name,
    )


def semantic_coverage_metrics(result: SemanticValidationResult) -> dict[str, Any]:
    return {
        "semantic_checked": True,
        "semantic_model": result.model_name,
        "semantic_support_avg": round(float(result.support_avg), 4) if result.support_avg is not None else None,
        "semantic_support_min": round(float(result.support_min), 4) if result.support_min is not None else None,
        "semantic_contradiction_max": round(float(result.contradiction_max), 4)
        if result.contradiction_max is not None
        else None,
        "semantic_lines_below_threshold": int(result.lines_below_threshold),
        "semantic_lines_contradiction": int(result.lines_contradiction),
        "semantic_status": result.status,
    }
