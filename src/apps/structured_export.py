"""Structured export pipeline for research reports.

Renders a :class:`ResearchReport` (and its companion :class:`EvidencePack`)
into **BibTeX**, **LaTeX**, or **Markdown** bibliography formats.  When a
paper-database path is supplied the exporter resolves raw ``paper_id`` values
to full bibliographic metadata (authors, title, year, venue, DOI …).
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Sequence

from src.core.schemas import EvidencePack, PaperRecord, ResearchReport

log = logging.getLogger(__name__)

ExportFormat = Literal["bibtex", "latex", "markdown"]
SUPPORTED_FORMATS: tuple[ExportFormat, ...] = ("bibtex", "latex", "markdown")

# ---------------------------------------------------------------------------
# Paper metadata resolution
# ---------------------------------------------------------------------------


@dataclass
class PaperMeta:
    """Lightweight container for resolved bibliographic metadata."""

    paper_id: str
    title: str = ""
    authors: list[str] = field(default_factory=list)
    year: int | None = None
    venue: str | None = None
    doi: str | None = None
    arxiv_id: str | None = None
    url: str = ""


def _load_papers_from_db(db_path: str) -> dict[str, PaperMeta]:
    """Load all papers from the SQLite DB into id→PaperMeta map."""
    try:
        with sqlite3.connect(db_path) as conn:
            rows = conn.execute(
                "SELECT paper_id, title, authors, year, venue, doi, arxiv_id, url FROM papers"
            ).fetchall()
    except (sqlite3.OperationalError, sqlite3.DatabaseError) as exc:
        log.warning("Could not read papers DB %s: %s", db_path, exc)
        return {}
    out: dict[str, PaperMeta] = {}
    for r in rows:
        out[r[0]] = PaperMeta(
            paper_id=r[0],
            title=r[1] or "",
            authors=r[2].split("|") if r[2] else [],
            year=r[3],
            venue=r[4],
            doi=r[5],
            arxiv_id=r[6],
            url=str(r[7] or ""),
        )
    return out


def _resolve_paper_ids(
    report: ResearchReport,
    evidence: EvidencePack | None,
    papers_db_path: str | None,
) -> dict[str, PaperMeta]:
    """Collect every unique paper_id from the report & evidence, then resolve."""
    ids: set[str] = set()
    for item in report.shortlist:
        ids.add(item.paper_id)
    if evidence:
        for item in evidence.items:
            ids.add(item.paper_id)
    # citation strings look like "(Pdoi_10_1007_...:S11)" — extract paper_id
    for cit in report.citations:
        pid = _extract_paper_id_from_citation(cit)
        if pid:
            ids.add(pid)

    db_papers: dict[str, PaperMeta] = {}
    if papers_db_path and Path(papers_db_path).exists():
        db_papers = _load_papers_from_db(papers_db_path)

    resolved: dict[str, PaperMeta] = {}
    for pid in sorted(ids):
        if pid in db_papers:
            resolved[pid] = db_papers[pid]
        else:
            resolved[pid] = PaperMeta(paper_id=pid, title=pid)
    return resolved


_CIT_RE = re.compile(r"\(?(P[^:)]+)")


def _extract_paper_id_from_citation(cit: str) -> str | None:
    """Extract a paper_id from a citation string like ``(Pdoi_10_...:S11)``."""
    m = _CIT_RE.search(cit)
    if not m:
        return None
    raw = m.group(1)
    # Reverse the sanitisation: Pdoi_10_1007_... → doi:10.1007/...
    if raw.startswith("Pdoi_"):
        parts = raw[5:].split("_")
        # First two segments are always the DOI prefix (e.g. "10" and "1007")
        if len(parts) >= 3:
            return "doi:" + parts[0] + "." + "/".join(parts[1:])
    return raw


# ---------------------------------------------------------------------------
# BibTeX key derivation
# ---------------------------------------------------------------------------

def _bibtex_key(meta: PaperMeta) -> str:
    """Generate a stable BibTeX cite-key from metadata."""
    surname = ""
    if meta.authors:
        # Take the last word of the first author's name
        surname = re.sub(r"[^a-zA-Z]", "", meta.authors[0].split()[-1]).lower()
    year = str(meta.year) if meta.year else "nd"
    # First significant word of title (skip articles)
    title_word = ""
    for w in (meta.title or "").split():
        w_clean = re.sub(r"[^a-zA-Z]", "", w).lower()
        if w_clean and w_clean not in {"a", "an", "the", "on", "of", "in", "for", "and", "to"}:
            title_word = w_clean
            break
    return f"{surname}{year}{title_word}" or meta.paper_id.replace(":", "_").replace("/", "_")


# ---------------------------------------------------------------------------
# Format renderers
# ---------------------------------------------------------------------------

def _render_bibtex(
    papers: dict[str, PaperMeta],
    report: ResearchReport,
) -> str:
    """Render a BibTeX bibliography string."""
    entries: list[str] = []
    for pid, meta in papers.items():
        key = _bibtex_key(meta)
        lines = [f"@article{{{key},"]
        if meta.title:
            lines.append(f"  title     = {{{meta.title}}},")
        if meta.authors:
            lines.append(f"  author    = {{{' and '.join(meta.authors)}}},")
        if meta.year:
            lines.append(f"  year      = {{{meta.year}}},")
        if meta.venue:
            lines.append(f"  journal   = {{{meta.venue}}},")
        if meta.doi:
            lines.append(f"  doi       = {{{meta.doi}}},")
        if meta.arxiv_id:
            lines.append(f"  eprint    = {{{meta.arxiv_id}}},")
            lines.append("  archiveprefix = {arXiv},")
        if meta.url:
            lines.append(f"  url       = {{{meta.url}}},")
        lines.append("}")
        entries.append("\n".join(lines))
    return "\n\n".join(entries) + "\n"


def _render_latex(
    papers: dict[str, PaperMeta],
    report: ResearchReport,
    evidence: EvidencePack | None,
) -> str:
    r"""Render a self-contained LaTeX document with the report and bibliography.

    Uses ``\begin{thebibliography}`` so no external .bib file is needed.
    """
    lines: list[str] = [
        r"\documentclass{article}",
        r"\usepackage[utf8]{inputenc}",
        r"\usepackage{hyperref}",
        r"\usepackage{geometry}",
        r"\geometry{margin=1in}",
        "",
        r"\title{Research Report}",
        r"\date{}",
        "",
        r"\begin{document}",
        r"\maketitle",
        "",
        r"\section*{Question}",
        _latex_escape(report.question),
        "",
    ]

    # Shortlist
    if report.shortlist:
        lines.append(r"\section*{Paper Shortlist}")
        lines.append(r"\begin{itemize}")
        for item in report.shortlist:
            meta = papers.get(item.paper_id)
            label = meta.title if meta and meta.title != meta.paper_id else item.paper_id
            lines.append(rf"  \item \textbf{{{_latex_escape(label)}}}: {_latex_escape(item.reason)}")
        lines.append(r"\end{itemize}")
        lines.append("")

    # Synthesis
    lines.append(r"\section*{Synthesis}")
    lines.append(_latex_escape(report.synthesis))
    lines.append("")

    # Key claims
    if report.key_claims:
        lines.append(r"\section*{Key Claims}")
        lines.append(r"\begin{itemize}")
        for c in report.key_claims:
            lines.append(rf"  \item {_latex_escape(c)}")
        lines.append(r"\end{itemize}")
        lines.append("")

    # Gaps
    if report.gaps:
        lines.append(r"\section*{Gaps}")
        lines.append(r"\begin{itemize}")
        for g in report.gaps:
            lines.append(rf"  \item {_latex_escape(g)}")
        lines.append(r"\end{itemize}")
        lines.append("")

    # Experiments
    if report.experiments:
        lines.append(r"\section*{Experiment Proposals}")
        lines.append(r"\begin{itemize}")
        for e in report.experiments:
            lines.append(rf"  \item {_latex_escape(e.proposal)}")
        lines.append(r"\end{itemize}")
        lines.append("")

    # Bibliography
    bib_items = _build_latex_bib_items(papers)
    if bib_items:
        lines.append(rf"\begin{{thebibliography}}{{{len(bib_items)}}}")
        lines.extend(bib_items)
        lines.append(r"\end{thebibliography}")
        lines.append("")

    lines.append(r"\end{document}")
    return "\n".join(lines) + "\n"


_LATEX_SPECIAL = re.compile(r"([#$%&_{}~^\\])")


def _latex_escape(text: str) -> str:
    """Escape LaTeX special characters in *text*."""
    def _repl(m: re.Match[str]) -> str:
        ch = m.group(1)
        if ch == "\\":
            return r"\textbackslash{}"
        if ch == "~":
            return r"\textasciitilde{}"
        if ch == "^":
            return r"\textasciicircum{}"
        return "\\" + ch
    return _LATEX_SPECIAL.sub(_repl, text)


def _build_latex_bib_items(papers: dict[str, PaperMeta]) -> list[str]:
    items: list[str] = []
    for pid, meta in papers.items():
        key = _bibtex_key(meta)
        author_str = ", ".join(meta.authors) if meta.authors else "Unknown"
        year_str = f" ({meta.year})" if meta.year else ""
        title_str = _latex_escape(meta.title or pid)
        venue_str = f", \\textit{{{_latex_escape(meta.venue)}}}" if meta.venue else ""
        doi_str = f", \\doi{{{meta.doi}}}" if meta.doi else ""
        items.append(
            rf"  \bibitem{{{key}}} {_latex_escape(author_str)}{year_str}. "
            rf"{title_str}{venue_str}{doi_str}."
        )
    return items


def _render_markdown_bib(
    papers: dict[str, PaperMeta],
    report: ResearchReport,
) -> str:
    """Render a Markdown bibliography / references section."""
    lines: list[str] = [
        "# References",
        "",
    ]
    for i, (pid, meta) in enumerate(papers.items(), 1):
        author_str = ", ".join(meta.authors) if meta.authors else "Unknown"
        year_str = f" ({meta.year})" if meta.year else ""
        title_str = meta.title or pid
        venue_str = f" *{meta.venue}*." if meta.venue else ""
        doi_link = f" [doi:{meta.doi}](https://doi.org/{meta.doi})" if meta.doi else ""
        url_str = f" [{meta.url}]({meta.url})" if meta.url and not meta.doi else ""
        lines.append(f"{i}. {author_str}{year_str}. **{title_str}**.{venue_str}{doi_link}{url_str}")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def export_report(
    report: ResearchReport,
    evidence: EvidencePack | None = None,
    *,
    fmt: ExportFormat = "bibtex",
    papers_db_path: str | None = None,
) -> str:
    """Export *report* in the requested format.

    Parameters
    ----------
    report:
        The research report to export.
    evidence:
        Optional evidence pack for richer output.
    fmt:
        Target format — ``"bibtex"``, ``"latex"``, or ``"markdown"``.
    papers_db_path:
        Path to the papers SQLite DB for full metadata resolution.
        When ``None`` the export falls back to raw paper IDs.

    Returns
    -------
    str
        The rendered export document.
    """
    if fmt not in SUPPORTED_FORMATS:
        raise ValueError(f"Unsupported export format {fmt!r}; choose from {SUPPORTED_FORMATS}")

    papers = _resolve_paper_ids(report, evidence, papers_db_path)
    log.info("Exporting report (%d papers) as %s", len(papers), fmt)

    if fmt == "bibtex":
        return _render_bibtex(papers, report)
    if fmt == "latex":
        return _render_latex(papers, report, evidence)
    # markdown
    return _render_markdown_bib(papers, report)


def export_report_to_file(
    report: ResearchReport,
    out_path: str,
    evidence: EvidencePack | None = None,
    *,
    fmt: ExportFormat = "bibtex",
    papers_db_path: str | None = None,
) -> str:
    """Export *report* and write to *out_path*.  Returns the path written."""
    content = export_report(report, evidence, fmt=fmt, papers_db_path=papers_db_path)
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    log.info("Wrote %s export to %s (%d bytes)", fmt, out_path, len(content))
    return str(p)


def load_report_json(path: str) -> ResearchReport:
    """Load a ResearchReport from a ``.json`` file."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return ResearchReport.model_validate(data)


def load_evidence_json(path: str) -> EvidencePack:
    """Load an EvidencePack from an ``.evidence.json`` file."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return EvidencePack.model_validate(data)


# ---------------------------------------------------------------------------
# Convenience: multi-format batch
# ---------------------------------------------------------------------------

def export_report_multi(
    report: ResearchReport,
    out_dir: str,
    basename: str = "report",
    evidence: EvidencePack | None = None,
    *,
    formats: Sequence[ExportFormat] = SUPPORTED_FORMATS,
    papers_db_path: str | None = None,
) -> dict[ExportFormat, str]:
    """Export into every requested format under *out_dir*.

    Returns a mapping of format → written file path.
    """
    results: dict[ExportFormat, str] = {}
    suffixes: dict[ExportFormat, str] = {
        "bibtex": ".bib",
        "latex": ".tex",
        "markdown": ".md",
    }
    for f in formats:
        suffix = suffixes[f]
        path = str(Path(out_dir) / f"{basename}{suffix}")
        export_report_to_file(report, path, evidence, fmt=f, papers_db_path=papers_db_path)
        results[f] = path
    return results
