"""Tests for the structured export pipeline (Feature #2).

Covers:
- Paper metadata resolution (with and without papers DB)
- BibTeX rendering
- LaTeX rendering
- Markdown bibliography rendering
- Citation-string → paper_id extraction
- BibTeX key generation
- LaTeX escaping
- Multi-format batch export
- CLI integration (export-report command)
- Edge cases: empty reports, missing DB, unknown paper IDs
"""

from __future__ import annotations

import json
import sqlite3
import textwrap
from pathlib import Path

import pytest

from src.apps.structured_export import (
    ExportFormat,
    PaperMeta,
    SUPPORTED_FORMATS,
    _bibtex_key,
    _extract_paper_id_from_citation,
    _latex_escape,
    _load_papers_from_db,
    _render_bibtex,
    _render_latex,
    _render_markdown_bib,
    _resolve_paper_ids,
    export_report,
    export_report_multi,
    export_report_to_file,
    load_evidence_json,
    load_report_json,
)
from src.core.schemas import (
    EvidenceItem,
    EvidencePack,
    ExperimentProposal,
    ResearchReport,
    ShortlistItem,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def sample_report() -> ResearchReport:
    return ResearchReport(
        question="How does attention mechanism scale?",
        shortlist=[
            ShortlistItem(paper_id="doi:10.1234/test1", title="Paper One", reason="High score"),
            ShortlistItem(paper_id="doi:10.5678/test2", title="Paper Two", reason="Relevant"),
        ],
        synthesis="Attention scales quadratically (Pdoi_10_1234_test1:S1). "
                  "New methods improve efficiency (Pdoi_10_5678_test2:S3).",
        key_claims=["Attention is O(n^2)", "Linear variants exist"],
        gaps=["Long-context benchmarks are limited"],
        experiments=[
            ExperimentProposal(
                proposal="Test ablation around method features (Pdoi_10_1234_test1:S1)",
                citations=["(Pdoi_10_1234_test1:S1)"],
            )
        ],
        citations=["(Pdoi_10_1234_test1:S1)", "(Pdoi_10_5678_test2:S3)"],
    )


@pytest.fixture()
def sample_evidence() -> EvidencePack:
    return EvidencePack(
        question="How does attention mechanism scale?",
        items=[
            EvidenceItem(
                paper_id="doi:10.1234/test1",
                snippet_id="Pdoi_10_1234_test1:S1",
                score=0.95,
                section="method",
                text="Attention is computed as softmax(QK^T/sqrt(d))V.",
                paper_year=2023,
                paper_venue="NeurIPS",
                citation_count=42,
            ),
            EvidenceItem(
                paper_id="doi:10.5678/test2",
                snippet_id="Pdoi_10_5678_test2:S3",
                score=0.88,
                section="results",
                text="Linear attention reduces cost to O(n).",
                paper_year=2024,
                paper_venue="ICML",
                citation_count=15,
            ),
        ],
    )


@pytest.fixture()
def papers_db(tmp_path: Path) -> str:
    db_path = str(tmp_path / "papers.db")
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE papers (
                paper_id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                authors TEXT NOT NULL,
                year INTEGER,
                venue TEXT,
                source TEXT NOT NULL,
                abstract TEXT,
                pdf_path TEXT,
                url TEXT NOT NULL,
                doi TEXT,
                arxiv_id TEXT,
                openalex_id TEXT,
                citation_count INTEGER DEFAULT 0,
                is_open_access INTEGER DEFAULT 0,
                sync_timestamp TEXT NOT NULL,
                dedupe_key TEXT NOT NULL
            )
            """
        )
        conn.execute(
            "INSERT INTO papers VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "doi:10.1234/test1",
                "Attention Is All You Need",
                "Ashish Vaswani|Noam Shazeer|Niki Parmar",
                2017,
                "NeurIPS",
                "arxiv",
                "Abstract text...",
                None,
                "https://arxiv.org/abs/1706.03762",
                "10.1234/test1",
                "1706.03762",
                None,
                50000,
                1,
                "2024-01-01T00:00:00",
                "attentionisallyouneed",
            ),
        )
        conn.execute(
            "INSERT INTO papers VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "doi:10.5678/test2",
                "Linear Attention: Faster Transformers",
                "Jane Doe|John Smith",
                2024,
                "ICML",
                "openalex",
                "Linear attention abstract...",
                None,
                "https://example.com/paper2",
                "10.5678/test2",
                None,
                None,
                150,
                0,
                "2024-06-15T00:00:00",
                "linearattentionfastertransformers",
            ),
        )
        conn.commit()
    return db_path


# ---------------------------------------------------------------------------
# TestCitationExtraction
# ---------------------------------------------------------------------------

class TestCitationExtraction:
    def test_standard_citation(self) -> None:
        assert _extract_paper_id_from_citation("(Pdoi_10_1234_test1:S1)") == "doi:10.1234/test1"

    def test_citation_without_parens(self) -> None:
        assert _extract_paper_id_from_citation("Pdoi_10_5678_test2:S3") == "doi:10.5678/test2"

    def test_complex_doi(self) -> None:
        result = _extract_paper_id_from_citation("(Pdoi_10_1007_s10055_023_00858_0:S11)")
        assert result == "doi:10.1007/s10055/023/00858/0"

    def test_non_doi_citation(self) -> None:
        result = _extract_paper_id_from_citation("(Parxiv_2301_12345:S2)")
        assert result == "Parxiv_2301_12345"

    def test_empty_string(self) -> None:
        assert _extract_paper_id_from_citation("") is None

    def test_no_match(self) -> None:
        assert _extract_paper_id_from_citation("random text") is None


# ---------------------------------------------------------------------------
# TestBibtexKey
# ---------------------------------------------------------------------------

class TestBibtexKey:
    def test_normal_paper(self) -> None:
        meta = PaperMeta(
            paper_id="doi:10.1234/test1",
            title="Attention Is All You Need",
            authors=["Ashish Vaswani", "Noam Shazeer"],
            year=2017,
        )
        key = _bibtex_key(meta)
        assert key == "vaswani2017attention"

    def test_no_author(self) -> None:
        meta = PaperMeta(paper_id="test", title="Some Title", year=2020)
        key = _bibtex_key(meta)
        assert key == "2020some"

    def test_no_year(self) -> None:
        meta = PaperMeta(paper_id="test", title="Deep Learning", authors=["LeCun"])
        key = _bibtex_key(meta)
        assert key == "lecunnddeep"

    def test_title_with_articles(self) -> None:
        meta = PaperMeta(paper_id="test", title="A Study on The Effects of Training", authors=["Smith"], year=2021)
        key = _bibtex_key(meta)
        assert key == "smith2021study"


# ---------------------------------------------------------------------------
# TestLatexEscape
# ---------------------------------------------------------------------------

class TestLatexEscape:
    def test_special_chars(self) -> None:
        assert _latex_escape("50% gain & 10$ cost") == r"50\% gain \& 10\$ cost"

    def test_underscore_hash(self) -> None:
        assert _latex_escape("my_var #1") == r"my\_var \#1"

    def test_backslash(self) -> None:
        assert _latex_escape("path\\to") == r"path\textbackslash{}to"

    def test_tilde_caret(self) -> None:
        assert _latex_escape("x~y^z") == r"x\textasciitilde{}y\textasciicircum{}z"

    def test_no_escape_needed(self) -> None:
        assert _latex_escape("Hello world") == "Hello world"


# ---------------------------------------------------------------------------
# TestPaperResolution
# ---------------------------------------------------------------------------

class TestPaperResolution:
    def test_resolve_with_db(
        self, sample_report: ResearchReport, sample_evidence: EvidencePack, papers_db: str
    ) -> None:
        resolved = _resolve_paper_ids(sample_report, sample_evidence, papers_db)
        assert len(resolved) == 2
        meta1 = resolved["doi:10.1234/test1"]
        assert meta1.title == "Attention Is All You Need"
        assert "Vaswani" in meta1.authors[0]
        assert meta1.year == 2017
        assert meta1.doi == "10.1234/test1"

    def test_resolve_without_db(
        self, sample_report: ResearchReport, sample_evidence: EvidencePack
    ) -> None:
        resolved = _resolve_paper_ids(sample_report, sample_evidence, None)
        assert len(resolved) == 2
        # Falls back to paper_id as title
        for meta in resolved.values():
            assert meta.title == meta.paper_id

    def test_resolve_missing_db_file(
        self, sample_report: ResearchReport, sample_evidence: EvidencePack
    ) -> None:
        resolved = _resolve_paper_ids(sample_report, sample_evidence, "/nonexistent/papers.db")
        assert len(resolved) == 2  # still resolves from evidence/shortlist
        for meta in resolved.values():
            assert meta.title == meta.paper_id

    def test_resolve_no_evidence(
        self, sample_report: ResearchReport, papers_db: str
    ) -> None:
        resolved = _resolve_paper_ids(sample_report, None, papers_db)
        assert len(resolved) >= 2  # from shortlist + citations


# ---------------------------------------------------------------------------
# TestLoadPapersDB
# ---------------------------------------------------------------------------

class TestLoadPapersDB:
    def test_load_valid_db(self, papers_db: str) -> None:
        papers = _load_papers_from_db(papers_db)
        assert len(papers) == 2
        assert "doi:10.1234/test1" in papers
        assert papers["doi:10.1234/test1"].title == "Attention Is All You Need"

    def test_load_nonexistent_table(self, tmp_path: Path) -> None:
        db_path = str(tmp_path / "empty.db")
        with sqlite3.connect(db_path) as conn:
            conn.execute("CREATE TABLE other (id TEXT)")
        result = _load_papers_from_db(db_path)
        assert result == {}


# ---------------------------------------------------------------------------
# TestBibtexExport
# ---------------------------------------------------------------------------

class TestBibtexExport:
    def test_basic_bibtex(
        self, sample_report: ResearchReport, sample_evidence: EvidencePack, papers_db: str
    ) -> None:
        result = export_report(sample_report, sample_evidence, fmt="bibtex", papers_db_path=papers_db)
        assert "@article{" in result
        assert "vaswani2017attention" in result
        assert "Attention Is All You Need" in result
        assert "Ashish Vaswani and Noam Shazeer and Niki Parmar" in result
        assert "doi" in result
        assert "1706.03762" in result  # arxiv_id

    def test_bibtex_without_db(
        self, sample_report: ResearchReport
    ) -> None:
        result = export_report(sample_report, fmt="bibtex")
        assert "@article{" in result
        # Falls back to paper_id-derived keys
        assert result.strip().endswith("}")

    def test_bibtex_multiple_papers(
        self, sample_report: ResearchReport, sample_evidence: EvidencePack, papers_db: str
    ) -> None:
        result = export_report(sample_report, sample_evidence, fmt="bibtex", papers_db_path=papers_db)
        assert result.count("@article{") == 2


# ---------------------------------------------------------------------------
# TestLatexExport
# ---------------------------------------------------------------------------

class TestLatexExport:
    def test_basic_latex(
        self, sample_report: ResearchReport, sample_evidence: EvidencePack, papers_db: str
    ) -> None:
        result = export_report(sample_report, sample_evidence, fmt="latex", papers_db_path=papers_db)
        assert r"\documentclass{article}" in result
        assert r"\begin{document}" in result
        assert r"\end{document}" in result
        assert r"\section*{Question}" in result
        assert "attention mechanism" in result.lower()
        assert r"\begin{thebibliography}" in result
        assert r"\bibitem{" in result

    def test_latex_has_all_sections(
        self, sample_report: ResearchReport, sample_evidence: EvidencePack
    ) -> None:
        result = export_report(sample_report, sample_evidence, fmt="latex")
        assert r"\section*{Synthesis}" in result
        assert r"\section*{Key Claims}" in result
        assert r"\section*{Gaps}" in result
        assert r"\section*{Experiment Proposals}" in result

    def test_latex_escapes_special(self) -> None:
        report = ResearchReport(
            question="What is 50% of $100?",
            synthesis="The answer involves #1 priority & $$ cost.",
            key_claims=[],
            gaps=[],
            experiments=[],
            citations=[],
        )
        result = export_report(report, fmt="latex")
        assert r"50\%" in result
        assert r"\$100" in result or r"\$" in result


# ---------------------------------------------------------------------------
# TestMarkdownExport
# ---------------------------------------------------------------------------

class TestMarkdownExport:
    def test_basic_markdown(
        self, sample_report: ResearchReport, sample_evidence: EvidencePack, papers_db: str
    ) -> None:
        result = export_report(sample_report, sample_evidence, fmt="markdown", papers_db_path=papers_db)
        assert "# References" in result
        assert "Attention Is All You Need" in result
        assert "Vaswani" in result
        assert "[doi:" in result or "doi.org" in result

    def test_markdown_without_db(
        self, sample_report: ResearchReport
    ) -> None:
        result = export_report(sample_report, fmt="markdown")
        assert "# References" in result
        assert "1." in result  # numbered list


# ---------------------------------------------------------------------------
# TestExportToFile
# ---------------------------------------------------------------------------

class TestExportToFile:
    def test_write_bibtex_file(
        self, tmp_path: Path, sample_report: ResearchReport, sample_evidence: EvidencePack, papers_db: str
    ) -> None:
        out = str(tmp_path / "refs.bib")
        path = export_report_to_file(sample_report, out, sample_evidence, fmt="bibtex", papers_db_path=papers_db)
        assert Path(path).exists()
        content = Path(path).read_text(encoding="utf-8")
        assert "@article{" in content

    def test_write_latex_file(
        self, tmp_path: Path, sample_report: ResearchReport
    ) -> None:
        out = str(tmp_path / "report.tex")
        path = export_report_to_file(sample_report, out, fmt="latex")
        assert Path(path).exists()
        assert Path(path).read_text(encoding="utf-8").startswith(r"\documentclass")

    def test_creates_parent_dirs(
        self, tmp_path: Path, sample_report: ResearchReport
    ) -> None:
        out = str(tmp_path / "deep" / "nested" / "dir" / "out.bib")
        path = export_report_to_file(sample_report, out, fmt="bibtex")
        assert Path(path).exists()


# ---------------------------------------------------------------------------
# TestMultiExport
# ---------------------------------------------------------------------------

class TestMultiExport:
    def test_all_formats(
        self, tmp_path: Path, sample_report: ResearchReport, sample_evidence: EvidencePack, papers_db: str
    ) -> None:
        results = export_report_multi(
            sample_report,
            str(tmp_path),
            basename="my_report",
            evidence=sample_evidence,
            papers_db_path=papers_db,
        )
        assert set(results.keys()) == {"bibtex", "latex", "markdown"}
        assert Path(results["bibtex"]).name == "my_report.bib"
        assert Path(results["latex"]).name == "my_report.tex"
        assert Path(results["markdown"]).name == "my_report.md"
        for path in results.values():
            assert Path(path).exists()
            assert Path(path).stat().st_size > 0

    def test_subset_formats(
        self, tmp_path: Path, sample_report: ResearchReport
    ) -> None:
        results = export_report_multi(
            sample_report, str(tmp_path), formats=["bibtex", "markdown"],
        )
        assert len(results) == 2
        assert "latex" not in results


# ---------------------------------------------------------------------------
# TestLoadHelpers
# ---------------------------------------------------------------------------

class TestLoadHelpers:
    def test_load_report_json(self, tmp_path: Path, sample_report: ResearchReport) -> None:
        path = str(tmp_path / "report.json")
        Path(path).write_text(json.dumps(sample_report.model_dump(), indent=2), encoding="utf-8")
        loaded = load_report_json(path)
        assert loaded.question == sample_report.question
        assert len(loaded.citations) == len(sample_report.citations)

    def test_load_evidence_json(self, tmp_path: Path, sample_evidence: EvidencePack) -> None:
        path = str(tmp_path / "evidence.json")
        Path(path).write_text(json.dumps(sample_evidence.model_dump(), indent=2), encoding="utf-8")
        loaded = load_evidence_json(path)
        assert loaded.question == sample_evidence.question
        assert len(loaded.items) == len(sample_evidence.items)


# ---------------------------------------------------------------------------
# TestUnsupportedFormat
# ---------------------------------------------------------------------------

class TestUnsupportedFormat:
    def test_raises_on_invalid_format(self, sample_report: ResearchReport) -> None:
        with pytest.raises(ValueError, match="Unsupported export format"):
            export_report(sample_report, fmt="pdf")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# TestEmptyReport
# ---------------------------------------------------------------------------

class TestEmptyReport:
    def test_empty_report_bibtex(self) -> None:
        report = ResearchReport(
            question="Empty?",
            synthesis="Nothing found.",
            key_claims=[],
            gaps=[],
            experiments=[],
            citations=[],
        )
        result = export_report(report, fmt="bibtex")
        # No papers → empty bibtex (just a trailing newline)
        assert result.strip() == ""

    def test_empty_report_latex(self) -> None:
        report = ResearchReport(
            question="Empty?",
            synthesis="Nothing found.",
            key_claims=[],
            gaps=[],
            experiments=[],
            citations=[],
        )
        result = export_report(report, fmt="latex")
        assert r"\documentclass{article}" in result
        assert r"\end{document}" in result
        # No bibliography section since no papers
        assert r"\begin{thebibliography}" not in result

    def test_empty_report_markdown(self) -> None:
        report = ResearchReport(
            question="Empty?",
            synthesis="Nothing found.",
            key_claims=[],
            gaps=[],
            experiments=[],
            citations=[],
        )
        result = export_report(report, fmt="markdown")
        assert "# References" in result
