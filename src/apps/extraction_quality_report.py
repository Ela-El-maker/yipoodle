from __future__ import annotations

from pathlib import Path
import json


def _sample_text(snippets: list[dict], max_chars: int = 180) -> str:
    for sn in snippets:
        txt = str(sn.get("text", "")).strip()
        if txt:
            return txt[:max_chars]
    return ""


def build_extraction_quality_report(corpus_dir: str) -> dict[str, object]:
    files = sorted(Path(corpus_dir).glob("*.json"))
    papers: list[dict[str, object]] = []

    total_pages = 0
    total_empty_pages = 0
    ocr_papers = 0
    layout_table_pages = 0
    layout_footnote_pages = 0
    layout_fallback_papers = 0

    for fp in files:
        payload = json.loads(fp.read_text(encoding="utf-8"))
        paper = payload.get("paper", {})
        snippets = payload.get("snippets", [])
        meta = payload.get("extraction_meta", {}) or {}
        page_stats = meta.get("page_stats", []) or []
        pages_total = int(meta.get("pages_total", len(page_stats) or 0))
        if pages_total <= 0 and page_stats:
            pages_total = len(page_stats)
        empty_pages = int(meta.get("empty_pages", sum(1 for p in page_stats if p.get("empty"))))
        empty_page_pct = float(meta.get("empty_page_pct", (empty_pages / pages_total) if pages_total else 0.0))
        ocr_applied = bool(meta.get("ocr_applied", False))
        if ocr_applied:
            ocr_papers += 1
        if int((meta.get("layout_region_counts") or {}).get("table", 0) or 0) > 0:
            layout_table_pages += 1
        if int((meta.get("layout_region_counts") or {}).get("footnote", 0) or 0) > 0:
            layout_footnote_pages += 1
        if str(meta.get("layout_engine_used", "")) == "legacy" and meta.get("layout_confidence") is not None:
            layout_fallback_papers += 1
        total_pages += pages_total
        total_empty_pages += empty_pages

        worst_pages = sorted(
            (
                {"page": int(p.get("page", 0)), "chars": int(p.get("chars", 0)), "empty": bool(p.get("empty", False))}
                for p in page_stats
            ),
            key=lambda x: (x["chars"], x["page"]),
        )[:3]

        papers.append(
            {
                "paper_id": paper.get("paper_id", fp.stem),
                "extractor": meta.get("extractor"),
                "quality_score": meta.get("quality_score"),
                "quality_band": meta.get("quality_band"),
                "ocr_applied": ocr_applied,
                "layout_engine_used": meta.get("layout_engine_used"),
                "layout_confidence": meta.get("layout_confidence"),
                "layout_region_counts": meta.get("layout_region_counts", {}),
                "pages_total": pages_total,
                "empty_pages": empty_pages,
                "empty_page_pct": round(empty_page_pct * 100.0, 2),
                "snippet_count": len(snippets),
                "sample_text": _sample_text(snippets),
                "worst_pages": worst_pages,
            }
        )

    worst_by_quality = sorted(
        papers,
        key=lambda p: (float(p.get("quality_score") if p.get("quality_score") is not None else 2.0), -float(p.get("empty_page_pct", 0.0))),
    )[:10]
    worst_by_empty_pages = sorted(papers, key=lambda p: float(p.get("empty_page_pct", 0.0)), reverse=True)[:10]

    return {
        "summary": {
            "papers": len(papers),
            "total_pages": total_pages,
            "empty_pages": total_empty_pages,
            "empty_page_pct": round((total_empty_pages / total_pages) * 100.0, 2) if total_pages > 0 else 0.0,
            "ocr_papers": ocr_papers,
            "ocr_papers_pct": round((ocr_papers / len(papers)) * 100.0, 2) if papers else 0.0,
            "layout_table_papers": layout_table_pages,
            "layout_table_papers_pct": round((layout_table_pages / len(papers)) * 100.0, 2) if papers else 0.0,
            "layout_footnote_papers": layout_footnote_pages,
            "layout_footnote_papers_pct": round((layout_footnote_pages / len(papers)) * 100.0, 2) if papers else 0.0,
            "layout_fallback_papers": layout_fallback_papers,
        },
        "worst_by_quality": worst_by_quality,
        "worst_by_empty_pages": worst_by_empty_pages,
        "papers": papers,
    }


def render_extraction_quality_markdown(report: dict[str, object]) -> str:
    summary = report.get("summary", {})
    lines = [
        "# Extraction Quality Report",
        "",
        "## Summary",
        f"- papers: {summary.get('papers')}",
        f"- total_pages: {summary.get('total_pages')}",
        f"- empty_pages: {summary.get('empty_pages')}",
        f"- empty_page_pct: {summary.get('empty_page_pct')}%",
        f"- ocr_papers: {summary.get('ocr_papers')}",
        f"- ocr_papers_pct: {summary.get('ocr_papers_pct')}%",
        f"- layout_table_papers: {summary.get('layout_table_papers')}",
        f"- layout_table_papers_pct: {summary.get('layout_table_papers_pct')}%",
        f"- layout_footnote_papers: {summary.get('layout_footnote_papers')}",
        f"- layout_footnote_papers_pct: {summary.get('layout_footnote_papers_pct')}%",
        f"- layout_fallback_papers: {summary.get('layout_fallback_papers')}",
        "",
        "## Worst by Quality",
    ]
    for row in report.get("worst_by_quality", [])[:10]:
        lines.append(
            f"- `{row.get('paper_id')}` quality={row.get('quality_score')} band={row.get('quality_band')} "
            f"empty_page_pct={row.get('empty_page_pct')}% ocr={row.get('ocr_applied')}"
        )
    lines.extend(["", "## Per-PDF Stats"])
    for row in report.get("papers", []):
        lines.append(
            f"- `{row.get('paper_id')}` pages={row.get('pages_total')} empty={row.get('empty_pages')} "
            f"({row.get('empty_page_pct')}%) ocr={row.get('ocr_applied')} "
            f"layout={row.get('layout_engine_used')} conf={row.get('layout_confidence')} "
            f"quality={row.get('quality_score')}/{row.get('quality_band')}"
        )
        sample = str(row.get("sample_text", ""))
        if sample:
            lines.append(f"  sample: {sample}")
        worst = row.get("worst_pages", [])
        if worst:
            lines.append(f"  worst_pages: {worst}")
        if row.get("layout_region_counts"):
            lines.append(f"  layout_regions: {row.get('layout_region_counts')}")
    return "\n".join(lines) + "\n"


def write_extraction_quality_report(corpus_dir: str, out_path: str) -> str:
    report = build_extraction_quality_report(corpus_dir)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_extraction_quality_markdown(report), encoding="utf-8")
    out.with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return str(out)
