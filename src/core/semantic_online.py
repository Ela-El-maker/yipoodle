from __future__ import annotations

from dataclasses import dataclass
import json
import os
import re
from typing import Any

import requests

from src.core.schemas import EvidencePack, ResearchReport

_CIT_RE = re.compile(r"\(P[^:()\s]+:S\d+\)")


@dataclass
class OnlineSemanticResult:
    checked_lines: int
    support_avg: float | None
    support_min: float | None
    contradiction_max: float | None
    lines_below_threshold: int
    lines_contradiction: int
    status: str
    latency_ms: float
    error: str | None = None


def _extract_citations(text: str) -> list[str]:
    return _CIT_RE.findall(text)


def _extract_json_object(text: str) -> dict[str, Any] | None:
    text = text.strip()
    if not text:
        return None
    try:
        payload = json.loads(text)
        return payload if isinstance(payload, dict) else None
    except Exception:
        pass
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        payload = json.loads(m.group(0))
        return payload if isinstance(payload, dict) else None
    except Exception:
        return None


def _judge_line_online(
    *,
    claim: str,
    evidence_text: str,
    model: str,
    timeout_sec: float,
    base_url: str,
    api_key: str,
) -> tuple[float, float]:
    system = (
        "You are a strict research-faithfulness judge. "
        "Return ONLY JSON with keys support_score and contradiction_score in [0,1]."
    )
    user = (
        "Claim:\n"
        f"{claim}\n\n"
        "Evidence:\n"
        f"{evidence_text}\n\n"
        "Score semantic support and contradiction risk. "
        "Return JSON exactly: {\"support_score\": <float>, \"contradiction_score\": <float>}."
    )
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "temperature": 0.0,
        "response_format": {"type": "json_object"},
    }
    resp = requests.post(base_url.rstrip("/") + "/chat/completions", headers=headers, json=payload, timeout=timeout_sec)
    resp.raise_for_status()
    data = resp.json()
    content = (
        (((data.get("choices") or [{}])[0].get("message") or {}).get("content"))
        if isinstance(data, dict)
        else None
    )
    parsed = _extract_json_object(str(content or ""))
    if not parsed:
        raise RuntimeError("online_semantic_invalid_response")
    s = float(parsed.get("support_score"))
    c = float(parsed.get("contradiction_score"))
    s = max(0.0, min(1.0, s))
    c = max(0.0, min(1.0, c))
    return s, c


def validate_online_semantic_claim_support(
    report: ResearchReport,
    evidence: EvidencePack,
    *,
    model: str,
    min_support: float,
    max_contradiction: float,
    timeout_sec: float = 12.0,
    max_checks: int = 12,
    on_warn_only: bool = True,
    offline_line_status: dict[int, str] | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
) -> OnlineSemanticResult:
    import time

    base_url = base_url or os.getenv("DEEPSEEK_BASE_URL") or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    api_key = api_key or os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        raise RuntimeError("online_semantic_missing_api_key")

    evidence_by_cit = {f"({item.snippet_id})": item.text for item in evidence.items}
    lines = [ln.strip() for ln in report.synthesis.splitlines() if ln.strip()]
    supports: list[float] = []
    contradictions: list[float] = []
    lines_below = 0
    lines_contra = 0
    checked = 0

    t0 = time.perf_counter()
    for i, line in enumerate(lines, start=1):
        citations = _extract_citations(line)
        if not citations:
            continue
        if on_warn_only and offline_line_status and offline_line_status.get(i) == "supported":
            continue
        claim = _CIT_RE.sub("", line).strip()
        if not claim:
            continue
        ev = " ".join(evidence_by_cit.get(c, "") for c in citations).strip()
        if not ev:
            continue
        support, contradiction = _judge_line_online(
            claim=claim,
            evidence_text=ev,
            model=model,
            timeout_sec=timeout_sec,
            base_url=base_url,
            api_key=api_key,
        )
        checked += 1
        supports.append(support)
        contradictions.append(contradiction)
        if support < float(min_support):
            lines_below += 1
        if contradiction > float(max_contradiction):
            lines_contra += 1
        if checked >= int(max_checks):
            break

    latency_ms = (time.perf_counter() - t0) * 1000.0
    status = "pass" if lines_below == 0 and lines_contra == 0 else "warn"
    return OnlineSemanticResult(
        checked_lines=checked,
        support_avg=(sum(supports) / len(supports)) if supports else None,
        support_min=min(supports) if supports else None,
        contradiction_max=max(contradictions) if contradictions else None,
        lines_below_threshold=lines_below,
        lines_contradiction=lines_contra,
        status=status,
        latency_ms=latency_ms,
        error=None,
    )
