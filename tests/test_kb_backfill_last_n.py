import json

from src.apps.kb_ingest import backfill_kb
from src.core.schemas import EvidenceItem, EvidencePack, ResearchReport


def _mk_bundle(root, idx: int) -> None:
    sid = f"P{idx}:S1"
    citation = f"({sid})"
    claim = f"model drift risk {idx}"
    report = ResearchReport(
        question="q",
        shortlist=[],
        synthesis=f"Claim: {claim} {citation}",
        key_claims=[claim],
        gaps=[],
        experiments=[],
        citations=[citation],
    )
    evidence = EvidencePack(
        question="q",
        items=[EvidenceItem(paper_id=f"P{idx}", snippet_id=sid, score=0.6, section="body", text=claim)],
    )
    md = "\n".join([
        "# Research Report", "", "## Synthesis", report.synthesis, "", "## Key Claims", f"- {claim} {citation}", "", "## Citations", f"- {citation}"
    ])
    p = root / f"r{idx}.md"
    p.write_text(md, encoding="utf-8")
    p.with_suffix(".json").write_text(json.dumps(report.model_dump(), indent=2), encoding="utf-8")
    p.with_suffix(".evidence.json").write_text(json.dumps(evidence.model_dump(), indent=2), encoding="utf-8")


def test_kb_backfill_last_n(tmp_path) -> None:
    reports = tmp_path / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    for i in range(3):
        _mk_bundle(reports, i)

    db = tmp_path / "kb.db"
    out = backfill_kb(kb_db=str(db), reports_dir=str(reports), topic="finance", last_n=2)
    assert out["kb_backfill_attempted"] == 2
    assert out["kb_backfill_succeeded"] >= 1
