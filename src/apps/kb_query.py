from __future__ import annotations

from typing import Any

from src.apps.kb_store import connect_kb, init_kb, query_claims
from src.core.schemas import EvidenceItem


def run_kb_query(*, kb_db: str, query: str, topic: str | None, top_k: int = 10) -> dict[str, Any]:
    init_kb(kb_db)
    with connect_kb(kb_db) as conn:
        rows = query_claims(conn, query=query, topic=topic, top_k=int(top_k))
    return {
        "kb_db": kb_db,
        "query": query,
        "topic": topic,
        "top_k": int(top_k),
        "count": len(rows),
        "claims": rows,
    }


def _paper_id_from_snippet_id(snippet_id: str) -> str:
    sid = str(snippet_id)
    if ":S" in sid:
        return sid.split(":S", 1)[0]
    return "KB"


def query_kb_as_evidence(
    *,
    kb_db: str,
    question: str,
    topic: str | None,
    top_k: int,
    merge_weight: float,
) -> list[EvidenceItem]:
    payload = run_kb_query(kb_db=kb_db, query=question, topic=topic, top_k=top_k)
    out: list[EvidenceItem] = []
    for claim in payload.get("claims", []):
        claim_text = str(claim.get("claim_text") or "").strip()
        conf = float(claim.get("confidence") or 0.0)
        if not claim_text:
            continue
        ev = claim.get("evidence", []) or []
        if not ev:
            continue
        for item in ev[:3]:
            sid = str(item.get("snippet_id") or "").strip()
            if not sid:
                continue
            source_type = str(item.get("source_type") or "paper")
            venue = "kb_live" if source_type == "live" else "kb_paper"
            support = float(item.get("support_score") or 0.0)
            score = float(merge_weight) * max(0.0, min(1.0, conf)) * max(0.0, min(1.0, support or 1.0))
            out.append(
                EvidenceItem(
                    paper_id=_paper_id_from_snippet_id(sid),
                    snippet_id=sid,
                    score=score,
                    section="kb_claim",
                    text=claim_text,
                    paper_year=None,
                    paper_venue=venue,
                    citation_count=0,
                    extraction_quality_score=None,
                    extraction_quality_band=None,
                    extraction_source=None,
                )
            )
    out.sort(key=lambda x: x.score, reverse=True)
    return out[: int(top_k)]
