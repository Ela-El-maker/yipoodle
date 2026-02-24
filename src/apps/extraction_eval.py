from __future__ import annotations

from pathlib import Path
import json
import re
from typing import Any


def _compact(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def _load_corpus(corpus_dir: str) -> dict[str, dict[str, object]]:
    out: dict[str, dict[str, object]] = {}
    for fp in sorted(Path(corpus_dir).glob("*.json")):
        payload = json.loads(fp.read_text(encoding="utf-8"))
        paper = payload.get("paper", {}) or {}
        paper_id = str(paper.get("paper_id", fp.stem))
        out[paper_id] = payload
    return out


def _resolve_payload_for_gold_row(
    corpus: dict[str, dict[str, object]],
    row: dict[str, object],
) -> tuple[str | None, dict[str, object] | None]:
    paper_id = str(row.get("paper_id", "")).strip()
    if paper_id and paper_id in corpus:
        return paper_id, corpus.get(paper_id)

    wanted_doi = str(row.get("doi", "")).strip().lower()
    wanted_arxiv = str(row.get("arxiv_id", "")).strip().lower()
    wanted_title = _compact(str(row.get("title", "")))
    for pid, payload in corpus.items():
        paper = payload.get("paper", {}) or {}
        cand_doi = str(paper.get("doi", "")).strip().lower()
        cand_arxiv = str(paper.get("arxiv_id", "")).strip().lower()
        cand_title = _compact(str(paper.get("title", "")))
        if wanted_doi and cand_doi and wanted_doi == cand_doi:
            return pid, payload
        if wanted_arxiv and cand_arxiv and wanted_arxiv == cand_arxiv:
            return pid, payload
        if wanted_title and cand_title and wanted_title in cand_title:
            return pid, payload
    return None, None


def _paper_text(payload: dict[str, object]) -> str:
    snippets = payload.get("snippets", []) or []
    return _compact(" ".join(str(sn.get("text", "")) for sn in snippets))


def _check_contains(text: str, needle: str) -> tuple[bool, str]:
    n = _compact(needle)
    ok = bool(n and n in text)
    return ok, f"contains:{needle}"


def _check_ordered_contains(text: str, needles: list[str]) -> tuple[bool, str]:
    pos = -1
    for n in needles:
        k = _compact(n)
        if not k:
            continue
        nxt = text.find(k, pos + 1)
        if nxt < 0:
            return False, f"ordered_contains_missing:{n}"
        pos = nxt
    return True, "ordered_contains"


def _check_min_chars(text: str, min_chars: int) -> tuple[bool, str]:
    ok = len(text) >= int(min_chars)
    return ok, f"min_chars:{len(text)}>={int(min_chars)}"


def _check_page_nonempty_ratio(payload: dict[str, object], min_ratio: float) -> tuple[bool, str]:
    meta = payload.get("extraction_meta", {}) or {}
    page_stats = meta.get("page_stats", []) or []
    if not page_stats:
        return False, "page_nonempty_ratio_missing_page_stats"
    total = len(page_stats)
    nonempty = sum(1 for p in page_stats if not bool(p.get("empty", False)))
    ratio = (nonempty / total) if total > 0 else 0.0
    ok = ratio >= float(min_ratio)
    return ok, f"page_nonempty_ratio:{ratio:.3f}>={float(min_ratio):.3f}"


def evaluate_extraction_against_gold(corpus_dir: str, gold_path: str) -> dict[str, object]:
    corpus = _load_corpus(corpus_dir)
    gold = json.loads(Path(gold_path).read_text(encoding="utf-8"))
    papers = gold.get("papers", []) or []
    results: list[dict[str, object]] = []
    total_weight = 0.0
    passed_weight = 0.0

    for row in papers:
        paper_id = str(row.get("paper_id", ""))
        checks = row.get("checks", []) or []
        matched_paper_id, payload = _resolve_payload_for_gold_row(corpus, row)
        if payload is None:
            paper_checks = [
                {
                    "id": str(ch.get("id", "")),
                    "type": str(ch.get("type", "")),
                    "weight": float(ch.get("weight", 1.0)),
                    "passed": False,
                    "detail": "paper_missing_in_corpus",
                }
                for ch in checks
            ]
            for ch in paper_checks:
                total_weight += float(ch["weight"])
            results.append({"paper_id": paper_id, "matched_paper_id": None, "checks": paper_checks})
            continue

        text = _paper_text(payload)
        paper_checks: list[dict[str, object]] = []
        for ch in checks:
            cid = str(ch.get("id", ""))
            ctype = str(ch.get("type", "contains"))
            weight = float(ch.get("weight", 1.0))
            passed = False
            detail = "unknown_check_type"
            if ctype == "contains":
                passed, detail = _check_contains(text, str(ch.get("needle", "")))
            elif ctype == "ordered_contains":
                passed, detail = _check_ordered_contains(text, [str(x) for x in (ch.get("needles", []) or [])])
            elif ctype == "min_chars":
                passed, detail = _check_min_chars(text, int(ch.get("min_chars", 0)))
            elif ctype == "page_nonempty_ratio":
                passed, detail = _check_page_nonempty_ratio(payload, float(ch.get("min_ratio", 0.0)))
            total_weight += weight
            if passed:
                passed_weight += weight
            paper_checks.append(
                {"id": cid, "type": ctype, "weight": weight, "passed": bool(passed), "detail": detail}
            )
        results.append({"paper_id": paper_id, "matched_paper_id": matched_paper_id or paper_id, "checks": paper_checks})

    score = (passed_weight / total_weight) if total_weight > 0 else 0.0
    passed_checks = sum(1 for p in results for c in p.get("checks", []) if bool(c.get("passed", False)))
    total_checks = sum(len(p.get("checks", [])) for p in results)
    return {
        "summary": {
            "papers_in_gold": len(papers),
            "papers_found_in_corpus": sum(
                1 for r in results if any(c.get("detail") != "paper_missing_in_corpus" for c in r["checks"])
            ),
            "total_checks": total_checks,
            "passed_checks": passed_checks,
            "weighted_score": round(score, 4),
        },
        "results": results,
    }


def render_extraction_eval_markdown(report: dict[str, object]) -> str:
    summary = report.get("summary", {}) or {}
    lines = [
        "# Extraction Eval Report",
        "",
        "## Summary",
        f"- papers_in_gold: {summary.get('papers_in_gold')}",
        f"- papers_found_in_corpus: {summary.get('papers_found_in_corpus')}",
        f"- total_checks: {summary.get('total_checks')}",
        f"- passed_checks: {summary.get('passed_checks')}",
        f"- weighted_score: {summary.get('weighted_score')}",
        "",
        "## Per-Paper Checks",
    ]
    for row in report.get("results", []) or []:
        lines.append(f"- `{row.get('paper_id')}`")
        for ch in row.get("checks", []) or []:
            lines.append(
                f"  - [{ 'PASS' if ch.get('passed') else 'FAIL' }] {ch.get('id')} ({ch.get('type')}, w={ch.get('weight')}): {ch.get('detail')}"
            )
    return "\n".join(lines) + "\n"


def write_extraction_eval_report(corpus_dir: str, gold_path: str, out_path: str) -> tuple[str, dict[str, object]]:
    report = evaluate_extraction_against_gold(corpus_dir, gold_path)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_extraction_eval_markdown(report), encoding="utf-8")
    out.with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return str(out), report


def scaffold_extraction_gold(
    corpus_dir: str,
    out_path: str,
    max_papers: int = 20,
    checks_per_paper: int = 2,
    min_chars: int = 500,
) -> str:
    corpus = _load_corpus(corpus_dir)
    papers: list[dict[str, Any]] = []
    for paper_id, payload in list(corpus.items())[: max(1, int(max_papers))]:
        snippets = payload.get("snippets", []) or []
        checks: list[dict[str, Any]] = []
        checks.append(
            {
                "id": "min_chars",
                "type": "min_chars",
                "min_chars": int(min_chars),
                "weight": 1.0,
            }
        )

        snippet_texts = [str(s.get("text", "")).strip() for s in snippets if str(s.get("text", "")).strip()]
        for idx, text in enumerate(snippet_texts[: max(0, int(checks_per_paper) - 1)]):
            compact = _compact(text)
            words = [w for w in compact.split(" ") if w]
            needle = " ".join(words[: min(6, len(words))]).strip()
            if not needle:
                continue
            checks.append(
                {
                    "id": f"contains_seed_{idx+1}",
                    "type": "contains",
                    "needle": needle,
                    "weight": 1.0,
                }
            )

        papers.append({"paper_id": paper_id, "checks": checks})

    payload = {"version": 1, "generated_from_corpus": corpus_dir, "papers": papers}
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return str(out)
