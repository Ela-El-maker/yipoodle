from __future__ import annotations

import re
from collections import Counter
from typing import Any

from src.core.schemas import EvidencePack, ResearchReport


CITATION_RE = re.compile(r"\((?:P[^:()\s]+:S\d+|SNAP:[A-Za-z0-9_.-]+:S\d+)\)")
NUMBER_RE = re.compile(r"\b\d+(?:\.\d+)?\b")


def extract_citations(text: str) -> list[str]:
    return CITATION_RE.findall(text)


def validate_citation_format(citation: str) -> bool:
    return bool(CITATION_RE.fullmatch(citation))


def validate_report_citations(report: ResearchReport) -> list[str]:
    errors: list[str] = []
    is_direct_answer = bool((report.retrieval_diagnostics or {}).get("direct_answer_used", False))
    all_text = [report.synthesis] + report.gaps + [e.proposal for e in report.experiments]
    joined = "\n".join(all_text)
    found = set(extract_citations(joined))

    if "Not found in sources." not in joined and not found and not is_direct_answer:
        errors.append("Report contains no citations")

    for c in report.citations:
        if not validate_citation_format(c):
            errors.append(f"Invalid citation format: {c}")

    used_missing = found.difference(set(report.citations))
    if used_missing:
        errors.append(f"Citations used but not listed in report.citations: {sorted(used_missing)}")

    for exp in report.experiments:
        if not exp.citations:
            errors.append("Experiment proposal missing citations")
    return errors


def validate_no_new_numbers(report_text: str, evidence: EvidencePack) -> list[str]:
    clean_report = CITATION_RE.sub("", report_text)
    evidence_numbers = Counter(NUMBER_RE.findall("\n".join(i.text for i in evidence.items)))
    report_numbers = Counter(NUMBER_RE.findall(clean_report))
    errors = []
    for n, count in report_numbers.items():
        if evidence_numbers[n] < count:
            errors.append(f"Number {n} appears {count}x in report but only {evidence_numbers[n]}x in evidence")
    return errors


def report_coverage_metrics(report: ResearchReport, evidence: EvidencePack) -> dict[str, float | int]:
    all_text = "\n".join([report.synthesis] + report.gaps + [e.proposal for e in report.experiments])
    claim_lines = [ln.strip() for ln in all_text.splitlines() if ln.strip()]
    cited_claim_lines = sum(1 for ln in claim_lines if extract_citations(ln))
    used_citations = set(extract_citations(all_text))
    evidence_citations = {f"({item.snippet_id})" for item in evidence.items}

    coverage = (cited_claim_lines / len(claim_lines)) if claim_lines else 1.0
    evidence_usage = (len(used_citations & evidence_citations) / len(evidence_citations)) if evidence_citations else 0.0
    return {
        "claim_lines": len(claim_lines),
        "cited_claim_lines": cited_claim_lines,
        "citation_coverage": round(coverage, 4),
        "evidence_items": len(evidence.items),
        "used_evidence_items": len(used_citations & evidence_citations),
        "evidence_usage": round(evidence_usage, 4),
    }


def _normalize_tokens(text: str) -> set[str]:
    return {tok.lower() for tok in re.findall(r"[a-zA-Z0-9]+", text) if len(tok) >= 3}


def validate_claim_support(report: ResearchReport, evidence: EvidencePack, min_overlap: float = 0.15, min_shared_tokens: int = 2) -> list[str]:
    evidence_by_cit = {f"({item.snippet_id})": item.text for item in evidence.items}
    lines = [ln.strip() for ln in report.synthesis.splitlines() if ln.strip()]
    errors: list[str] = []

    for i, line in enumerate(lines, start=1):
        if "insufficient evidence" in line.lower() or "not found in sources." in line.lower():
            continue
        citations = extract_citations(line)
        if not citations:
            continue
        claim = CITATION_RE.sub("", line)
        claim_toks = _normalize_tokens(claim)
        if not claim_toks:
            continue
        evidence_text = " ".join(evidence_by_cit.get(c, "") for c in citations)
        evidence_toks = _normalize_tokens(evidence_text)
        if not evidence_toks:
            errors.append(f"Line {i}: cited evidence not found in evidence pack")
            continue
        shared = len(claim_toks & evidence_toks)
        overlap = shared / max(1, len(claim_toks))
        if overlap < min_overlap or shared < min_shared_tokens:
            errors.append(
                f"Line {i}: low claim-evidence support (overlap={overlap:.3f}, shared={shared}, "
                f"required_overlap={min_overlap:.3f}, required_shared={min_shared_tokens})"
            )
    return errors


def validate_semantic_claim_support(
    report: ResearchReport,
    evidence: EvidencePack,
    *,
    semantic_mode: str = "offline",
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    min_support: float = 0.55,
    max_contradiction: float = 0.30,
    shadow_mode: bool = True,
    fail_on_low_support: bool = False,
    online_model: str = "gpt-4o-mini",
    online_timeout_sec: float = 12.0,
    online_max_checks: int = 12,
    online_on_warn_only: bool = True,
    online_base_url: str | None = None,
    online_api_key: str | None = None,
) -> tuple[list[str], dict[str, Any], list[str]]:
    """
    Semantic support wrapper used by CLI/research/automation.
    Returns: (errors, metrics, warnings)
    """
    if semantic_mode not in {"offline", "online", "hybrid"}:
        return [f"invalid_semantic_mode:{semantic_mode}"], {"semantic_checked": False}, []

    try:
        from src.core.semantic_validation import semantic_coverage_metrics, validate_semantic_claim_support as _sem_validate
    except Exception as exc:  # pragma: no cover - import environment edge case
        warning = f"semantic_check_unavailable: {exc}"
        return [], {"semantic_checked": False, "semantic_error": str(exc), "semantic_shadow_mode": shadow_mode}, [warning]

    offline_line_status: dict[int, str] = {}
    try:
        result = (
            _sem_validate(
                report,
                evidence,
                model_name=model_name,
                min_support=min_support,
                max_contradiction=max_contradiction,
            )
            if semantic_mode in {"offline", "hybrid"}
            else None
        )
        if result is not None:
            offline_line_status = {lr.line_no: lr.status for lr in result.line_results}
    except Exception as exc:
        warning = f"semantic_check_failed: {exc}"
        if shadow_mode or semantic_mode == "online":
            return (
                [],
                {"semantic_checked": False, "semantic_error": str(exc), "semantic_shadow_mode": shadow_mode},
                [warning],
            )
        return [warning], {"semantic_checked": False, "semantic_error": str(exc), "semantic_shadow_mode": shadow_mode}, []

    metrics = semantic_coverage_metrics(result) if result is not None else {"semantic_checked": False}
    metrics["semantic_shadow_mode"] = bool(shadow_mode)
    metrics["semantic_mode"] = semantic_mode
    metrics["semantic_min_support_threshold"] = float(min_support)
    metrics["semantic_max_contradiction_threshold"] = float(max_contradiction)

    # Optional online layer.
    if semantic_mode in {"online", "hybrid"}:
        try:
            from src.core.semantic_online import validate_online_semantic_claim_support

            online_result = validate_online_semantic_claim_support(
                report,
                evidence,
                model=online_model,
                min_support=min_support,
                max_contradiction=max_contradiction,
                timeout_sec=online_timeout_sec,
                max_checks=online_max_checks,
                on_warn_only=online_on_warn_only,
                offline_line_status=offline_line_status if semantic_mode == "hybrid" else None,
                base_url=online_base_url,
                api_key=online_api_key,
            )
            metrics.update(
                {
                    "online_semantic_checked": True,
                    "online_semantic_model": online_model,
                    "online_semantic_checked_lines": online_result.checked_lines,
                    "online_semantic_support_avg": round(float(online_result.support_avg), 4)
                    if online_result.support_avg is not None
                    else None,
                    "online_semantic_support_min": round(float(online_result.support_min), 4)
                    if online_result.support_min is not None
                    else None,
                    "online_semantic_contradiction_max": round(float(online_result.contradiction_max), 4)
                    if online_result.contradiction_max is not None
                    else None,
                    "online_semantic_lines_below_threshold": int(online_result.lines_below_threshold),
                    "online_semantic_lines_contradiction": int(online_result.lines_contradiction),
                    "online_semantic_status": online_result.status,
                    "online_semantic_latency_ms": round(float(online_result.latency_ms), 3),
                }
            )
            if semantic_mode == "online":
                metrics["semantic_checked"] = True
                metrics["semantic_model"] = online_model
                metrics["semantic_support_avg"] = metrics.get("online_semantic_support_avg")
                metrics["semantic_support_min"] = metrics.get("online_semantic_support_min")
                metrics["semantic_contradiction_max"] = metrics.get("online_semantic_contradiction_max")
                metrics["semantic_lines_below_threshold"] = metrics.get("online_semantic_lines_below_threshold")
                metrics["semantic_lines_contradiction"] = metrics.get("online_semantic_lines_contradiction")
                metrics["semantic_status"] = metrics.get("online_semantic_status")
        except Exception as exc:
            metrics.update({"online_semantic_checked": False, "online_semantic_error": str(exc)})
            if semantic_mode == "online":
                warn = f"online_semantic_unavailable_fallback:{exc}"
                if shadow_mode:
                    return [], metrics, [warn]
                return [warn], metrics, []

    warnings: list[str] = []
    errors: list[str] = []
    lines_below = int(metrics.get("semantic_lines_below_threshold") or 0)
    lines_contra = int(metrics.get("semantic_lines_contradiction") or 0)
    if lines_below > 0 or lines_contra > 0:
        if shadow_mode:
            metrics["semantic_status"] = "warn"
            warnings.append(
                "semantic_faithfulness_warn: "
                f"{lines_below} weak_support, {lines_contra} possible_contradiction"
            )
        elif fail_on_low_support:
            metrics["semantic_status"] = "fail"
            errors.append(
                "semantic_faithfulness_fail: "
                f"{lines_below} weak_support, {lines_contra} possible_contradiction"
            )
        else:
            metrics["semantic_status"] = "warn"
            warnings.append(
                "semantic_faithfulness_warn: "
                f"{lines_below} weak_support, {lines_contra} possible_contradiction"
            )
    return errors, metrics, warnings
