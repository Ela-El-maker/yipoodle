import json

from src.apps.kb_diff import run_kb_diff
from src.apps.kb_ingest import ingest_report_to_kb
from src.core.schemas import EvidenceItem, EvidencePack, ResearchReport


def _bundle(base, name: str, claim_text: str, sid: str) -> tuple[str, str, str]:
    citation = f"({sid})"
    report = ResearchReport(
        question="q",
        shortlist=[],
        synthesis=f"Claim: {claim_text} {citation}",
        key_claims=[claim_text],
        gaps=[],
        experiments=[],
        citations=[citation],
    )
    evidence = EvidencePack(
        question="q",
        items=[EvidenceItem(paper_id=sid.split(":", 1)[0], snippet_id=sid, score=0.7, section="body", text=claim_text)],
    )
    md = "\n".join([
        "# Research Report", "", "## Synthesis", report.synthesis, "", "## Key Claims", f"- {claim_text} {citation}", "", "## Citations", f"- {citation}"
    ])
    p = base / f"{name}.md"
    p.write_text(md, encoding="utf-8")
    p.with_suffix(".json").write_text(json.dumps(report.model_dump(), indent=2), encoding="utf-8")
    p.with_suffix(".evidence.json").write_text(json.dumps(evidence.model_dump(), indent=2), encoding="utf-8")
    p.with_suffix(".metrics.json").write_text(json.dumps({"citation_coverage": 1.0}, indent=2), encoding="utf-8")
    return str(p), str(p.with_suffix(".evidence.json")), str(p.with_suffix(".metrics.json"))


def test_kb_diff_since_run(tmp_path) -> None:
    db = tmp_path / "kb.db"
    r1, e1, m1 = _bundle(tmp_path, "r1", "transformers fail under drift", "P1:S1")
    ingest_report_to_kb(report_path=r1, evidence_path=e1, metrics_path=m1, kb_db=str(db), topic="finance", run_id="run1")

    r2, e2, m2 = _bundle(tmp_path, "r2", "attention models are sensitive to noise", "P2:S1")
    ingest_report_to_kb(report_path=r2, evidence_path=e2, metrics_path=m2, kb_db=str(db), topic="finance", run_id="run2")

    diff = run_kb_diff(kb_db=str(db), topic="finance", since_run="run1")
    assert diff["entries"]
    assert diff["aggregate_counts"]["added"] >= 1
