from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import json

from src.apps.kb_claim_parser import parse_key_claims
from src.apps.kb_ingest import ingest_report_to_kb
from src.apps.research_copilot import run_research
from src.core.schemas import ResearchReport
from src.core.validation import extract_citations


def _utc_ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _slug(text: str) -> str:
    out = "".join(ch.lower() if ch.isalnum() else "_" for ch in text.strip())
    out = "_".join(part for part in out.split("_") if part)
    return out or "notes"


def _dedup_keep_order(items: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for x in items:
        k = x.strip()
        if not k or k in seen:
            continue
        seen.add(k)
        out.append(k)
    return out


def _extract_cited_claim_candidates(rep: ResearchReport) -> list[str]:
    cands: list[str] = []
    for c in rep.key_claims:
        if extract_citations(c):
            cands.append(c.strip())
    if cands:
        return _dedup_keep_order(cands)

    for ln in rep.synthesis.splitlines():
        s = ln.strip()
        if not s:
            continue
        if s.lower().startswith("claim:"):
            s = s[6:].strip()
        if extract_citations(s):
            cands.append(s)
    return _dedup_keep_order(cands)


def _upsert_key_claims_section(markdown: str, claim_lines: list[str]) -> str:
    lines = (markdown or "").splitlines()
    if not lines:
        lines = ["# Research Report", ""]
    out: list[str] = []
    i = 0
    replaced = False
    while i < len(lines):
        ln = lines[i]
        if ln.strip().lower() == "## key claims":
            replaced = True
            out.append("## Key Claims")
            for c in claim_lines:
                out.append(f"- {c}")
            i += 1
            while i < len(lines) and not lines[i].strip().startswith("## "):
                i += 1
            continue
        out.append(ln)
        i += 1
    if not replaced:
        inserted = False
        for idx, ln in enumerate(out):
            if ln.strip().lower() == "## gaps":
                out = out[:idx] + ["## Key Claims", *[f"- {c}" for c in claim_lines], ""] + out[idx:]
                inserted = True
                break
        if not inserted:
            if out and out[-1].strip():
                out.append("")
            out.append("## Key Claims")
            for c in claim_lines:
                out.append(f"- {c}")
    return "\n".join(out) + "\n"


def run_notes_mode(
    *,
    question: str,
    index_path: str,
    kb_db: str,
    topic: str | None,
    out_path: str,
    sources_config_path: str | None = None,
) -> dict[str, Any]:
    ts = _utc_ts()
    slug = _slug(topic or question)[:80]
    research_out = Path("runs/notes") / f"{ts}_{slug}.research.md"
    report_path = run_research(
        index_path=index_path,
        question=question,
        top_k=8,
        out_path=str(research_out),
        min_items=2,
        min_score=0.1,
        retrieval_mode="lexical",
        sources_config_path=sources_config_path,
    )

    report_md = Path(report_path)
    report_json = report_md.with_suffix(".json")
    report_evidence = report_md.with_suffix(".evidence.json")
    report_metrics = report_md.with_suffix(".metrics.json")
    rep = ResearchReport(**json.loads(report_json.read_text(encoding="utf-8")))

    md_text = report_md.read_text(encoding="utf-8")
    claims = parse_key_claims(md_text)
    needs_normalize = (not claims) or any(not c.citations for c in claims)
    if needs_normalize:
        fallback_claims = _extract_cited_claim_candidates(rep)
        if not fallback_claims:
            raise ValueError("notes_mode_failed: missing cited key claims in report markdown/synthesis")
        normalized = _upsert_key_claims_section(md_text, fallback_claims)
        report_md.write_text(normalized, encoding="utf-8")
        md_text = normalized
        claims = parse_key_claims(md_text)
    if not claims or any(not c.citations for c in claims):
        raise ValueError("notes_mode_failed: every key claim must include citation(s)")

    ingest_stats = ingest_report_to_kb(
        report_path=str(report_md),
        evidence_path=str(report_evidence),
        metrics_path=str(report_metrics),
        kb_db=kb_db,
        topic=topic or slug,
        run_id=ts,
    )

    cited_claim_lines = _extract_cited_claim_candidates(rep)
    if not cited_claim_lines:
        cited_claim_lines = [f"{c.claim_text} {' '.join(c.citations)}".strip() for c in claims]
    lines = [
        "# Study Notes",
        "",
        f"## Question\n{question}",
        "",
        "## Summary",
        rep.synthesis,
        "",
        "## Key Claims",
    ]
    for c in cited_claim_lines:
        lines.append(f"- {c}")

    lines.extend(
        [
            "",
            "## Citations",
        ]
    )
    for c in sorted(set(rep.citations)):
        lines.append(f"- {c}")

    lines.extend(
        [
            "",
            "## Knowledge Links",
            f"- report_path: {report_md}",
            f"- evidence_path: {report_evidence}",
            f"- metrics_path: {report_metrics}",
            f"- kb_db: {kb_db}",
            f"- kb_ingest_stats: added={ingest_stats.get('kb_claims_added')}, updated={ingest_stats.get('kb_claims_updated')}, disputed={ingest_stats.get('kb_claims_disputed')}",
        ]
    )

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    sidecar = out.with_suffix(".json")
    payload = {
        "mode": "notes",
        "question": question,
        "topic": topic or slug,
        "notes_path": str(out),
        "notes_report_path": str(report_md),
        "kb_ingest_succeeded": bool(ingest_stats.get("kb_ingest_succeeded", False)),
        "kb_claims_added": int(ingest_stats.get("kb_claims_added", 0)),
        "kb_claims_updated": int(ingest_stats.get("kb_claims_updated", 0)),
    }
    sidecar.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload
