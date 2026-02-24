from __future__ import annotations

from pathlib import Path
import json
import re


def _compact(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def evaluate_corpus_health(
    corpus_dir: str,
    extract_stats_path: str | None = None,
    min_snippets: int = 1,
    min_avg_chars_per_paper: int = 500,
    min_avg_chars_per_page: int = 80,
    max_extract_error_rate: float = 0.8,
) -> dict[str, object]:
    root = Path(corpus_dir)
    files = sorted(root.glob("*.json")) if root.exists() else []
    paper_count = len(files)
    snippet_count = 0
    total_chars = 0
    total_pages = 0
    papers_with_page_stats = 0
    parse_errors = 0

    for p in files:
        try:
            payload = json.loads(p.read_text(encoding="utf-8"))
            snippets = payload.get("snippets", [])
            extraction_meta = payload.get("extraction_meta", {}) or {}
            snippet_count += len(snippets)
            for sn in snippets:
                total_chars += len(_compact(str(sn.get("text", ""))))
            pages_total = int(extraction_meta.get("pages_total", 0) or 0)
            if pages_total > 0:
                total_pages += pages_total
                papers_with_page_stats += 1
        except Exception:
            parse_errors += 1

    avg_chars_per_paper = (total_chars / paper_count) if paper_count > 0 else 0.0
    avg_chars_per_snippet = (total_chars / snippet_count) if snippet_count > 0 else 0.0
    avg_chars_per_page = (total_chars / total_pages) if total_pages > 0 else 0.0
    page_stats_coverage_pct = (papers_with_page_stats / paper_count * 100.0) if paper_count > 0 else 0.0

    extract_error_rate: float | None = None
    extract_stats: dict[str, object] | None = None
    if extract_stats_path and Path(extract_stats_path).exists():
        extract_stats = json.loads(Path(extract_stats_path).read_text(encoding="utf-8"))
        processed = int(extract_stats.get("processed", 0))
        if processed > 0:
            failed = int(extract_stats.get("failed_pdfs_count", 0))
            if failed == 0:
                failed = max(0, processed - int(extract_stats.get("created", 0)))
            extract_error_rate = failed / processed

    reasons: list[str] = []
    warnings: list[str] = []
    if snippet_count < min_snippets:
        reasons.append(f"snippet_count_below_threshold:{snippet_count}<{min_snippets}")
    if avg_chars_per_paper < float(min_avg_chars_per_paper):
        reasons.append(
            f"avg_chars_per_paper_below_threshold:{avg_chars_per_paper:.2f}<{float(min_avg_chars_per_paper):.2f}"
        )
    if avg_chars_per_page < float(min_avg_chars_per_page):
        reasons.append(
            f"avg_chars_per_page_below_threshold:{avg_chars_per_page:.2f}<{float(min_avg_chars_per_page):.2f}"
        )
    if extract_error_rate is not None and extract_error_rate > max_extract_error_rate:
        reasons.append(f"extract_error_rate_above_threshold:{extract_error_rate:.4f}>{max_extract_error_rate:.4f}")
    if paper_count > 0 and papers_with_page_stats == 0:
        warnings.append("page_stats_missing_for_corpus")

    healthy = len(reasons) == 0
    return {
        "healthy": healthy,
        "reasons": reasons,
        "warnings": warnings,
        "paper_count": paper_count,
        "papers_with_page_stats": papers_with_page_stats,
        "page_stats_coverage_pct": round(page_stats_coverage_pct, 2),
        "snippet_count": snippet_count,
        "total_chars": total_chars,
        "total_pages": total_pages,
        "avg_chars_per_paper": round(avg_chars_per_paper, 3),
        "avg_chars_per_page": round(avg_chars_per_page, 3),
        "avg_chars_per_snippet": round(avg_chars_per_snippet, 3),
        "parse_errors": parse_errors,
        "extract_error_rate": round(extract_error_rate, 6) if extract_error_rate is not None else None,
        "thresholds": {
            "min_snippets": min_snippets,
            "min_avg_chars_per_paper": min_avg_chars_per_paper,
            "min_avg_chars_per_page": min_avg_chars_per_page,
            "max_extract_error_rate": max_extract_error_rate,
        },
        "extract_stats_path": extract_stats_path,
    }
