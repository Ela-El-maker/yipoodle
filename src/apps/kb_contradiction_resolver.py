from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import json
import re
from typing import Any

from src.apps.kb_store import (
    connect_kb,
    get_topic_id,
    init_kb,
    insert_claim_version,
    record_change,
)
from src.apps.research_copilot import run_research
from src.apps.source_reliability import record_feedback, recompute_reliability
from src.core.schemas import EvidencePack


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _tokset(text: str) -> set[str]:
    return set(t for t in re.findall(r"[A-Za-z0-9]+", (text or "").lower()) if t)


def _norm_score(v: float) -> float:
    x = max(0.0, float(v))
    return x / (1.0 + x)


def _claim_support_from_evidence(claim_text: str, evidence: EvidencePack) -> float:
    cset = _tokset(claim_text)
    if not cset:
        return 0.0
    vals: list[float] = []
    for it in evidence.items:
        tset = _tokset(it.text)
        overlap = (len(cset & tset) / float(len(cset))) if cset else 0.0
        vals.append(0.7 * overlap + 0.3 * _norm_score(float(it.score)))
    if not vals:
        return 0.0
    vals = sorted(vals, reverse=True)[:3]
    return round(sum(vals) / float(len(vals)), 6)


def _claim_sources_from_evidence(
    *,
    papers_db: str | None,
    evidence: EvidencePack,
) -> set[str]:
    if not papers_db:
        return set()
    import sqlite3

    paper_ids = sorted(set(str(it.paper_id) for it in evidence.items if str(it.paper_id)))
    if not paper_ids:
        return set()
    qmarks = ",".join("?" for _ in paper_ids)
    sql = f"SELECT paper_id, source FROM papers WHERE paper_id IN ({qmarks})"
    out: set[str] = set()
    with sqlite3.connect(papers_db) as conn:
        for row in conn.execute(sql, paper_ids).fetchall():
            src = str(row[1] or "").strip().lower()
            if src:
                out.add(src)
    return out


def _fetch_disputed_pairs(conn, topic_id: int, max_pairs: int) -> list[dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT kc.id AS contradiction_id, kc.claim_id_a, kc.claim_id_b, kc.score, kc.reason, kc.detected_at,
               ca.claim_text AS claim_a_text, cb.claim_text AS claim_b_text,
               ca.status AS claim_a_status, cb.status AS claim_b_status
        FROM kb_contradiction kc
        JOIN kb_claim ca ON ca.id = kc.claim_id_a
        JOIN kb_claim cb ON cb.id = kc.claim_id_b
        WHERE ca.topic_id = ? AND cb.topic_id = ?
          AND ca.status = 'disputed' AND cb.status = 'disputed'
        ORDER BY kc.detected_at DESC
        LIMIT ?
        """,
        (int(topic_id), int(topic_id), int(max_pairs)),
    ).fetchall()
    seen: set[tuple[int, int]] = set()
    out: list[dict[str, Any]] = []
    for r in rows:
        a = int(r["claim_id_a"])
        b = int(r["claim_id_b"])
        key = (min(a, b), max(a, b))
        if key in seen:
            continue
        seen.add(key)
        out.append(
            {
                "contradiction_id": int(r["contradiction_id"]),
                "claim_id_a": a,
                "claim_id_b": b,
                "claim_a_text": str(r["claim_a_text"]),
                "claim_b_text": str(r["claim_b_text"]),
                "score": float(r["score"]),
                "reason": str(r["reason"]),
                "detected_at": str(r["detected_at"]),
            }
        )
    return out


def run_kb_contradiction_resolver(
    *,
    kb_db: str,
    topic: str,
    index_path: str,
    out_dir: str,
    run_id: str | None = None,
    max_pairs: int = 5,
    support_margin: float = 0.05,
    top_k: int = 8,
    min_items: int = 2,
    min_score: float = 0.0,
    retrieval_mode: str = "hybrid",
    alpha: float = 0.6,
    max_per_paper: int = 2,
    quality_prior_weight: float = 0.15,
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    sources_config_path: str | None = None,
    papers_db: str | None = None,
    reliability_db: str | None = None,
) -> dict[str, Any]:
    init_kb(kb_db)
    rid = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    now_iso = _utc_now()
    with connect_kb(kb_db) as conn:
        topic_id = get_topic_id(conn, topic)
        if topic_id is None:
            raise ValueError(f"topic not found in kb: {topic}")
        pairs = _fetch_disputed_pairs(conn, topic_id=topic_id, max_pairs=max_pairs)

    payload: dict[str, Any] = {
        "kb_db": kb_db,
        "topic": topic,
        "run_id": rid,
        "max_pairs": int(max_pairs),
        "support_margin": float(support_margin),
        "pairs_seen": len(pairs),
        "pairs_resolved": 0,
        "pairs_unresolved": 0,
        "pairs": [],
        "kb_change": {},
        "artifact_dir": str(out_root),
    }
    if not pairs:
        payload["status"] = "no_disputes"
        return payload

    changes = {"resolved": [], "unresolved": [], "counts": {"resolved": 0, "unresolved": 0}}
    source_feedback: dict[str, dict[str, int]] = {}

    with connect_kb(kb_db) as conn:
        conn.execute("BEGIN")
        for i, p in enumerate(pairs, start=1):
            q = (
                "Resolve this contradiction with strongest evidence. "
                f"Claim A: {p['claim_a_text']} "
                f"Claim B: {p['claim_b_text']}"
            )
            report_path = out_root / f"resolve_{i:02d}_{p['claim_id_a']}_{p['claim_id_b']}.md"
            run_research(
                index_path=index_path,
                question=q,
                top_k=top_k,
                out_path=str(report_path),
                min_items=min_items,
                min_score=min_score,
                retrieval_mode=retrieval_mode,
                alpha=alpha,
                max_per_paper=max_per_paper,
                quality_prior_weight=quality_prior_weight,
                embedding_model=embedding_model,
                sources_config_path=sources_config_path,
            )
            evidence = EvidencePack(**json.loads(report_path.with_suffix(".evidence.json").read_text(encoding="utf-8")))
            sa = _claim_support_from_evidence(str(p["claim_a_text"]), evidence)
            sb = _claim_support_from_evidence(str(p["claim_b_text"]), evidence)
            diff = round(abs(sa - sb), 6)

            row = {
                **p,
                "support_a": sa,
                "support_b": sb,
                "support_diff": diff,
                "resolution_report_path": str(report_path),
                "resolution_evidence_path": str(report_path.with_suffix(".evidence.json")),
                "resolution": "unresolved",
            }
            if diff < float(support_margin):
                payload["pairs_unresolved"] += 1
                changes["unresolved"].append({"claim_id_a": p["claim_id_a"], "claim_id_b": p["claim_id_b"], "reason": "low_margin"})
            else:
                if sa >= sb:
                    winner = int(p["claim_id_a"])
                    loser = int(p["claim_id_b"])
                    winner_support, loser_support = sa, sb
                else:
                    winner = int(p["claim_id_b"])
                    loser = int(p["claim_id_a"])
                    winner_support, loser_support = sb, sa
                conn.execute("UPDATE kb_claim SET status='active', last_seen_run_id=? WHERE id=?", (rid, winner))
                conn.execute("UPDATE kb_claim SET status='superseded', last_seen_run_id=? WHERE id=?", (rid, loser))
                insert_claim_version(
                    conn,
                    claim_id=winner,
                    claim_text=str(p["claim_a_text"] if winner == p["claim_id_a"] else p["claim_b_text"]),
                    confidence=min(1.0, max(0.0, 0.5 + winner_support / 2.0)),
                    run_id=rid,
                    reason=f"contradiction_resolved:winner;loser={loser};diff={diff}",
                    now_iso=now_iso,
                )
                insert_claim_version(
                    conn,
                    claim_id=loser,
                    claim_text=str(p["claim_a_text"] if loser == p["claim_id_a"] else p["claim_b_text"]),
                    confidence=min(1.0, max(0.0, 0.4 + loser_support / 3.0)),
                    run_id=rid,
                    reason=f"contradiction_resolved:loser;winner={winner};diff={diff}",
                    now_iso=now_iso,
                )
                row["resolution"] = "resolved"
                row["winner_claim_id"] = winner
                row["loser_claim_id"] = loser
                payload["pairs_resolved"] += 1
                changes["resolved"].append(
                    {"claim_id_a": p["claim_id_a"], "claim_id_b": p["claim_id_b"], "winner_claim_id": winner, "loser_claim_id": loser}
                )

                if reliability_db:
                    win_sources = _claim_sources_from_evidence(papers_db=papers_db, evidence=evidence)
                    for src in win_sources:
                        source_feedback.setdefault(src, {"confirmed": 0, "disputed": 0})
                        source_feedback[src]["confirmed"] += 1

            payload["pairs"].append(row)

        changes["counts"]["resolved"] = len(changes["resolved"])
        changes["counts"]["unresolved"] = len(changes["unresolved"])
        record_change(conn, topic_id=topic_id, run_id=rid, diff=changes, now_iso=now_iso)
        conn.commit()

    if reliability_db and source_feedback:
        updated_sources: list[str] = []
        for src, cnt in source_feedback.items():
            for _ in range(int(cnt.get("confirmed", 0))):
                record_feedback(reliability_db, src, "claim_confirmed", value=1.0, run_id=rid)
            for _ in range(int(cnt.get("disputed", 0))):
                record_feedback(reliability_db, src, "claim_disputed", value=1.0, run_id=rid)
            recompute_reliability(reliability_db, src)
            updated_sources.append(src)
        payload["reliability_sources_updated"] = sorted(updated_sources)
    else:
        payload["reliability_sources_updated"] = []

    payload["kb_change"] = changes
    payload["status"] = "ok"
    summary_path = out_root / "kb_contradiction_resolution.json"
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    payload["summary_path"] = str(summary_path)
    return payload
