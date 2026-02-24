import json

from src.apps.kb_ingest import ingest_report_to_kb
from src.apps.kb_query import run_kb_query
from src.core.schemas import EvidenceItem, EvidencePack, ResearchReport


def _write_report_bundle(base, *, claim_line: str, claim_bullet: str, citation: str) -> tuple[str, str, str]:
    report = ResearchReport(
        question="What are transformer risks?",
        shortlist=[],
        synthesis=claim_line,
        key_claims=[claim_bullet],
        gaps=[],
        experiments=[],
        citations=[citation],
    )
    evidence = EvidencePack(
        question=report.question,
        items=[
            EvidenceItem(
                paper_id="Pabc",
                snippet_id="Pabc:S1",
                score=0.8,
                section="body",
                text="Transformer models fail under distribution shift and market drift.",
            )
        ],
    )
    md = "\n".join(
        [
            "# Research Report",
            "",
            "## Question",
            report.question,
            "",
            "## Synthesis",
            report.synthesis,
            "",
            "## Key Claims",
            f"- {claim_bullet} {citation}",
            "",
            "## Gaps",
            "- none",
            "",
            "## Experiment Proposals",
            "",
            "## Citations",
            f"- {citation}",
        ]
    )
    md_path = base / "r.md"
    md_path.write_text(md, encoding="utf-8")
    md_path.with_suffix(".json").write_text(json.dumps(report.model_dump(), indent=2), encoding="utf-8")
    md_path.with_suffix(".evidence.json").write_text(json.dumps(evidence.model_dump(), indent=2), encoding="utf-8")
    md_path.with_suffix(".metrics.json").write_text(json.dumps({"citation_coverage": 1.0}, indent=2), encoding="utf-8")
    return str(md_path), str(md_path.with_suffix(".evidence.json")), str(md_path.with_suffix(".metrics.json"))


def test_kb_ingest_and_query(tmp_path) -> None:
    report_path, evidence_path, metrics_path = _write_report_bundle(
        tmp_path,
        claim_line="Claim: Transformer models fail under distribution shift. (Pabc:S1)",
        claim_bullet="Transformer models fail under distribution shift.",
        citation="(Pabc:S1)",
    )
    db = tmp_path / "kb.db"
    stats = ingest_report_to_kb(
        report_path=report_path,
        evidence_path=evidence_path,
        metrics_path=metrics_path,
        kb_db=str(db),
        topic="finance_markets",
        run_id="r1",
    )
    assert stats["kb_ingest_succeeded"] is True
    assert stats["kb_claims_added"] == 1

    payload = run_kb_query(kb_db=str(db), query="distribution shift", topic="finance_markets", top_k=5)
    assert payload["count"] >= 1
    assert payload["claims"][0]["evidence"][0]["snippet_id"] == "Pabc:S1"
