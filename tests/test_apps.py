from pathlib import Path

from src.apps.doc_writer import write_doc
from src.apps.evidence_extract import extract_snippets
from src.apps.index_builder import build_index
from src.apps.paper_ingest import dedupe_key_for_paper
from src.apps.research_copilot import build_research_report, run_research
from src.apps.release_notes import _bucket
from src.apps.retrieval import load_index
from src.apps.query_builder import build_query_plan
from src.core.schemas import EvidencePack, PaperRecord
from src.core.validation import extract_citations, validate_no_new_numbers


def test_dedupe_priority_keys() -> None:
    p1 = PaperRecord(
        paper_id="x1",
        title="A",
        authors=[],
        year=2024,
        venue=None,
        source="arxiv",
        abstract="",
        pdf_path=None,
        url="https://x",
        doi="10.1/abc",
        arxiv_id="1234.1",
        sync_timestamp="2026-01-01T00:00:00Z",
    )
    p2 = p1.model_copy(update={"doi": None})
    p3 = p1.model_copy(update={"doi": None, "arxiv_id": None, "title": "Title Based"})
    assert dedupe_key_for_paper(p1).startswith("doi:")
    assert dedupe_key_for_paper(p2).startswith("arxiv:")
    assert dedupe_key_for_paper(p3).startswith("title:")


def test_snippet_ids_stable() -> None:
    paper = PaperRecord(
        paper_id="arxiv:1",
        title="T",
        authors=[],
        year=2024,
        venue=None,
        source="arxiv",
        abstract="",
        pdf_path=None,
        url="https://x",
        doi=None,
        arxiv_id="1",
        sync_timestamp="2026-01-01T00:00:00Z",
    )
    text = "Paragraph one.\n\nParagraph two."
    s1 = extract_snippets(paper, text)
    s2 = extract_snippets(paper, text)
    assert [x.snippet_id for x in s1] == [x.snippet_id for x in s2]


def test_section_heading_detection() -> None:
    paper = PaperRecord(
        paper_id="arxiv:2",
        title="T",
        authors=[],
        year=2024,
        venue=None,
        source="arxiv",
        abstract="",
        pdf_path=None,
        url="https://x",
        doi=None,
        arxiv_id="2",
        sync_timestamp="2026-01-01T00:00:00Z",
    )
    text = "Abstract\nThis paper studies X.\n\nMethod\nWe train with Y.\n\nResults\nIt fails at night."
    snippets = extract_snippets(paper, text)
    sections = [s.section for s in snippets]
    assert "abstract" in sections
    assert "method" in sections
    assert "results" in sections


def test_doc_writer_and_release_bucket(tmp_path) -> None:
    out = write_doc("readme", "tests/fixtures/facts.yaml", str(tmp_path / "README.md"))
    content = Path(out).read_text(encoding="utf-8")
    assert "# Yipoodle Research Copilot" in content
    buckets = _bucket(["feat: add endpoint", "fix: retry", "BREAKING: old api removed"])
    assert "Added" in buckets and "Fixed" in buckets and "Breaking" in buckets


def test_validation_no_new_numbers() -> None:
    ev = EvidencePack(
        question="q",
        items=[
            {
                "paper_id": "p1",
                "snippet_id": "Pp1:S1",
                "score": 1.0,
                "section": "results",
                "text": "Model runs at 30 FPS.",
            }
        ],
    )
    errs = validate_no_new_numbers("We got 40 FPS (Pp1:S1)", ev)
    assert errs


def test_research_pipeline_from_fixture(tmp_path) -> None:
    corpus = "tests/fixtures/extracted"
    index_path = tmp_path / "index.json"
    stats = build_index(corpus, str(index_path))
    assert stats["snippets"] >= 2

    out = tmp_path / "report.md"
    run_research(str(index_path), "mobile segmentation limitations", 5, str(out))
    text = out.read_text(encoding="utf-8")
    cits = extract_citations(text)
    assert cits
    assert "## Synthesis" in text
    assert "## Coverage Metrics" in text
    assert out.with_suffix(".metrics.json").exists()


def test_insufficient_evidence_gate() -> None:
    evidence = EvidencePack(
        question="q",
        items=[{"paper_id": "p1", "snippet_id": "Pp1:S1", "score": 0.1, "section": "results", "text": "Weak mention only."}],
    )
    report = build_research_report("q", evidence, min_items=2, min_score=0.5)
    assert "Insufficient evidence" in report.synthesis


def test_retrieval_query_plan_and_diversification(tmp_path) -> None:
    index_path = tmp_path / "index.json"
    build_index("tests/fixtures/extracted", str(index_path))
    index = load_index(str(index_path))
    plan = build_query_plan("mobile segmentation limitations", domain="computer_vision")
    evidence = index.query(
        "mobile segmentation limitations",
        top_k=4,
        query_terms=plan.query_terms,
        term_boosts=plan.term_boosts,
        section_weights=plan.section_weights,
        max_per_paper=1,
    )
    assert evidence.items
    paper_ids = [i.paper_id for i in evidence.items]
    assert len(paper_ids) == len(set(paper_ids))
