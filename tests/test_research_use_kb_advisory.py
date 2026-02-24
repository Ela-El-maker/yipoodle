import json

from src.apps.kb_ingest import ingest_report_to_kb
from src.apps.research_copilot import run_research
from src.apps.retrieval import SimpleBM25Index, save_index
from src.core.schemas import EvidenceItem, EvidencePack, ResearchReport, SnippetRecord


def _seed_kb(tmp_path):
    report = ResearchReport(
        question="q",
        shortlist=[],
        synthesis="Claim: transformer models fail under drift. (Pabc:S1)",
        key_claims=["transformer models fail under drift"],
        gaps=[],
        experiments=[],
        citations=["(Pabc:S1)"],
    )
    evidence = EvidencePack(
        question="q",
        items=[
            EvidenceItem(
                paper_id="Pabc",
                snippet_id="Pabc:S1",
                score=0.8,
                section="body",
                text="Transformer models fail under drift in volatile markets.",
            )
        ],
    )
    md = "\n".join(
        [
            "# Research Report",
            "",
            "## Synthesis",
            report.synthesis,
            "",
            "## Key Claims",
            "- transformer models fail under drift. (Pabc:S1)",
            "",
            "## Citations",
            "- (Pabc:S1)",
        ]
    )
    p = tmp_path / "seed.md"
    p.write_text(md, encoding="utf-8")
    p.with_suffix(".json").write_text(json.dumps(report.model_dump(), indent=2), encoding="utf-8")
    p.with_suffix(".evidence.json").write_text(json.dumps(evidence.model_dump(), indent=2), encoding="utf-8")
    p.with_suffix(".metrics.json").write_text(json.dumps({"citation_coverage": 1.0}, indent=2), encoding="utf-8")
    db = tmp_path / "kb.db"
    ingest_report_to_kb(
        report_path=str(p),
        evidence_path=str(p.with_suffix(".evidence.json")),
        metrics_path=str(p.with_suffix(".metrics.json")),
        kb_db=str(db),
        topic="finance",
        run_id="seed",
    )
    return db


def test_research_use_kb_advisory(tmp_path) -> None:
    db = _seed_kb(tmp_path)

    idx = SimpleBM25Index.build(
        [
            SnippetRecord(
                snippet_id="Pz:S1",
                paper_id="Pz",
                section="body",
                text="unrelated content",
                token_count=2,
            )
        ]
    )
    index_path = tmp_path / "index.json"
    save_index(idx, str(index_path))

    out = tmp_path / "report.md"
    run_research(
        index_path=str(index_path),
        question="transformer drift risk",
        top_k=5,
        out_path=str(out),
        min_items=1,
        min_score=0.0,
        use_cache=False,
        use_kb=True,
        kb_db=str(db),
        kb_top_k=5,
        kb_merge_weight=0.4,
    )

    metrics = json.loads(out.with_suffix(".metrics.json").read_text(encoding="utf-8"))
    assert metrics["kb_query_used"] is True
    assert metrics["kb_candidates_injected"] >= 1

    report_md = out.read_text(encoding="utf-8")
    assert "(Pabc:S1)" in report_md
