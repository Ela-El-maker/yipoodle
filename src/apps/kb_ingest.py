from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import json
from typing import Any

from src.apps.kb_claim_parser import parse_key_claims
from src.apps.kb_confidence import canonical_hash, compute_claim_confidence
from src.apps.kb_store import (
    add_note,
    connect_kb,
    decay_topic_confidence,
    ensure_topic,
    find_potential_contradictions,
    init_kb,
    insert_claim_version,
    insert_contradiction,
    insert_evidence,
    mark_disputed,
    record_change,
    upsert_claim,
)
from src.core.schemas import EvidencePack, ResearchReport
from src.core.validation import (
    validate_claim_support,
    validate_no_new_numbers,
    validate_report_citations,
)


def _citation_to_snippet_id(citation: str) -> str:
    c = citation.strip()
    if c.startswith("(") and c.endswith(")"):
        c = c[1:-1]
    return c


def _norm_score(v: float) -> float:
    x = max(0.0, float(v))
    return x / (1.0 + x)


def _source_quality_from_snippet_id(snippet_id: str) -> float:
    return 0.78 if snippet_id.startswith("SNAP:") else 0.9


def ingest_report_to_kb(
    *,
    report_path: str,
    evidence_path: str,
    metrics_path: str | None,
    kb_db: str,
    topic: str,
    run_id: str | None = None,
) -> dict[str, Any]:
    init_kb(kb_db)
    rpt_md_path = Path(report_path)
    if not rpt_md_path.exists():
        raise FileNotFoundError(f"report not found: {report_path}")
    rpt_json_path = rpt_md_path.with_suffix(".json")
    if not rpt_json_path.exists():
        raise FileNotFoundError(f"report json sidecar required for ingest gate: {rpt_json_path}")

    report = ResearchReport(**json.loads(rpt_json_path.read_text(encoding="utf-8")))
    evidence = EvidencePack(**json.loads(Path(evidence_path).read_text(encoding="utf-8")))
    metrics = json.loads(Path(metrics_path).read_text(encoding="utf-8")) if metrics_path and Path(metrics_path).exists() else {}

    report_text = report.synthesis + "\n" + "\n".join(report.gaps + [e.proposal for e in report.experiments])
    errors = []
    errors.extend(validate_report_citations(report))
    errors.extend(validate_no_new_numbers(report_text, evidence))
    errors.extend(validate_claim_support(report, evidence))
    if errors:
        raise ValueError("kb_ingest_gate_failed: " + "; ".join(errors))

    md_text = rpt_md_path.read_text(encoding="utf-8")
    parsed_claims = parse_key_claims(md_text)
    if not parsed_claims:
        raise ValueError("kb_ingest_failed: missing or empty '## Key Claims' section")

    ev_by_cit = {f"({it.snippet_id})": it for it in evidence.items}
    now_iso = datetime.now(timezone.utc).isoformat()
    rid = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    stats: dict[str, Any] = {
        "kb_ingest_attempted": True,
        "kb_ingest_succeeded": False,
        "topic": topic,
        "run_id": rid,
        "kb_claims_seen": len(parsed_claims),
        "kb_claims_added": 0,
        "kb_claims_updated": 0,
        "kb_claims_disputed": 0,
        "kb_claims_skipped_no_citation": 0,
        "kb_claims_skipped_missing_evidence": 0,
        "kb_change_count": 0,
        "kb_change": {},
    }

    with connect_kb(kb_db) as conn:
        conn.execute("BEGIN")
        topic_id = ensure_topic(conn, topic)
        decay_topic_confidence(conn, topic_id=topic_id, now_iso=now_iso)

        added: list[str] = []
        updated: list[str] = []
        disputed: list[str] = []

        for claim in parsed_claims:
            if not claim.citations:
                stats["kb_claims_skipped_no_citation"] += 1
                continue
            cited_items = [ev_by_cit[c] for c in claim.citations if c in ev_by_cit]
            if not cited_items:
                stats["kb_claims_skipped_missing_evidence"] += 1
                continue

            mean_support = sum(_norm_score(float(i.score)) for i in cited_items) / max(1, len(cited_items))
            citation_quality = sum(_source_quality_from_snippet_id(i.snippet_id) for i in cited_items) / max(1, len(cited_items))
            recency = 1.0
            validation_signal = 1.0 if float(metrics.get("citation_coverage", 1.0)) >= 1.0 else 0.8
            conf = compute_claim_confidence(
                mean_support=mean_support,
                citation_quality=citation_quality,
                recency=recency,
                validation_signal=validation_signal,
            )

            chash = canonical_hash(claim.claim_text)
            claim_id, action, prev_conf, prev_status = upsert_claim(
                conn,
                topic_id=topic_id,
                claim_text=claim.claim_text,
                canonical_hash=chash,
                confidence=conf,
                run_id=rid,
                now_iso=now_iso,
            )
            if action == "added":
                stats["kb_claims_added"] += 1
                added.append(chash)
                reason = "added"
            else:
                stats["kb_claims_updated"] += 1
                updated.append(chash)
                reason = f"updated_conf:{prev_conf}->{conf};status:{prev_status}->active"

            insert_claim_version(
                conn,
                claim_id=claim_id,
                claim_text=claim.claim_text,
                confidence=conf,
                run_id=rid,
                reason=reason,
                now_iso=now_iso,
            )

            for c in claim.citations:
                item = ev_by_cit.get(c)
                if item is None:
                    continue
                sid = _citation_to_snippet_id(c)
                source_type = "live" if sid.startswith("SNAP:") else "paper"
                insert_evidence(
                    conn,
                    claim_id=claim_id,
                    snippet_id=sid,
                    source_type=source_type,
                    support_score=_norm_score(float(item.score)),
                    retrieved_at=now_iso,
                    run_id=rid,
                )

            contradictions = find_potential_contradictions(
                conn,
                topic_id=topic_id,
                claim_id=claim_id,
                claim_text=claim.claim_text,
            )
            for other_id, score, reason in contradictions:
                mark_disputed(conn, claim_id)
                mark_disputed(conn, other_id)
                insert_contradiction(
                    conn,
                    claim_id_a=claim_id,
                    claim_id_b=other_id,
                    score=score,
                    run_id=rid,
                    reason=reason,
                    now_iso=now_iso,
                )
                stats["kb_claims_disputed"] += 1
                disputed.append(chash)

        add_note(
            conn,
            topic_id=topic_id,
            note_path=str(rpt_md_path),
            run_id=rid,
            evidence_path=str(Path(evidence_path)),
            metrics_path=str(Path(metrics_path)) if metrics_path else None,
            now_iso=now_iso,
        )

        diff = {
            "added": sorted(set(added)),
            "updated": sorted(set(updated)),
            "disputed": sorted(set(disputed)),
            "superseded": [],
            "counts": {
                "added": len(set(added)),
                "updated": len(set(updated)),
                "disputed": len(set(disputed)),
                "superseded": 0,
            },
        }
        record_change(conn, topic_id=topic_id, run_id=rid, diff=diff, now_iso=now_iso)

        conn.commit()
        stats["kb_change"] = diff
        stats["kb_change_count"] = int(sum(diff["counts"].values()))
        stats["kb_ingest_succeeded"] = True
    return stats


def backfill_kb(
    *,
    kb_db: str,
    reports_dir: str,
    topic: str,
    last_n: int = 20,
) -> dict[str, Any]:
    root = Path(reports_dir)
    mds = sorted(root.rglob("*.md"), key=lambda p: p.stat().st_mtime, reverse=True)
    chosen = mds[: max(1, int(last_n))]
    out = {
        "kb_backfill_attempted": len(chosen),
        "kb_backfill_succeeded": 0,
        "kb_backfill_failed": 0,
        "topic": topic,
        "last_n": int(last_n),
        "files": [],
    }
    for md in chosen:
        ev = md.with_suffix(".evidence.json")
        met = md.with_suffix(".metrics.json")
        if not ev.exists():
            out["kb_backfill_failed"] += 1
            out["files"].append({"report": str(md), "ok": False, "error": "missing_evidence"})
            continue
        try:
            stats = ingest_report_to_kb(
                report_path=str(md),
                evidence_path=str(ev),
                metrics_path=str(met) if met.exists() else None,
                kb_db=kb_db,
                topic=topic,
            )
            out["kb_backfill_succeeded"] += 1
            out["files"].append({"report": str(md), "ok": True, "stats": stats})
        except Exception as exc:
            out["kb_backfill_failed"] += 1
            out["files"].append({"report": str(md), "ok": False, "error": str(exc)})
    return out
