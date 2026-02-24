from __future__ import annotations

from collections import Counter
from pathlib import Path
import json
import re
import sqlite3
import subprocess
import tempfile
import csv

from pypdf import PdfReader

from src.core.ids import snippet_id
from src.core.schemas import PaperRecord, SnippetRecord
from src.apps.layout_engine import REGION_KEYS, analyze_page_regions, reconstruct_page_text_v2

SECTION_HINTS = {
    "abstract": "abstract",
    "introduction": "introduction",
    "background": "background",
    "related work": "related_work",
    "method": "method",
    "approach": "method",
    "experiments": "experiments",
    "results": "results",
    "discussion": "discussion",
    "limitations": "limitations",
    "future work": "future_work",
    "conclusion": "conclusion",
}

FAILED_PDFS_CAP = 200
_OCR_STOPWORDS = {
    "eng": {"the", "and", "is", "are", "with", "for", "from", "this", "that", "using"},
    "spa": {"el", "la", "los", "las", "de", "que", "y", "en", "para", "con"},
    "deu": {"der", "die", "das", "und", "ist", "mit", "für", "von", "ein", "eine"},
    "fra": {"le", "la", "les", "de", "et", "est", "pour", "avec", "dans", "des"},
    "por": {"o", "a", "os", "as", "de", "e", "é", "para", "com", "em"},
    "ita": {"il", "lo", "la", "gli", "le", "di", "e", "per", "con", "nel"},
}


def _compact_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _extract_text_pypdf(pdf_path: str) -> tuple[str, list[dict[str, int | bool]]]:
    reader = PdfReader(pdf_path)
    texts = []
    page_stats: list[dict[str, int | bool]] = []
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        compact_len = len(_compact_text(text))
        texts.append(text)
        page_stats.append({"page": i, "chars": compact_len, "empty": compact_len == 0})
    return "\n".join(texts), page_stats


def _is_caption_like(text: str) -> bool:
    t = text.strip().lower()
    return t.startswith(("fig.", "figure ", "table ", "tab.", "algorithm "))


def _is_noise_block(text: str) -> bool:
    t = _compact_text(text)
    if not t:
        return True
    if len(t) <= 2:
        return True
    # Repeated symbol-heavy blocks are usually extraction artifacts.
    alnum = sum(1 for c in t if c.isalnum())
    if alnum == 0:
        return True
    if (alnum / max(1, len(t))) < 0.2:
        return True
    return False


def _should_drop_block_text(text: str) -> bool:
    t = _compact_text(text)
    if _is_noise_block(t):
        return True
    if _is_caption_like(t):
        return True
    return False


def _clean_blocks_for_layout(blocks: list[tuple], page_width: float, page_height: float) -> list[tuple[float, float, float, float, str]]:
    cleaned: list[tuple[float, float, float, float, str]] = []
    for block in blocks:
        x0, y0, x1, y1, text = block[:5]
        txt = str(text).strip()
        if _should_drop_block_text(txt):
            continue
        # Drop likely running headers/footers near page boundaries.
        if page_height > 0:
            near_top = float(y1) < page_height * 0.06
            near_bottom = float(y0) > page_height * 0.94
            if (near_top or near_bottom) and len(txt) < 120:
                continue
        # Tiny width/height blocks are usually junk.
        if (float(x1) - float(x0)) < max(8.0, page_width * 0.01):
            continue
        if (float(y1) - float(y0)) < 2.0:
            continue
        cleaned.append((float(x0), float(y0), float(x1), float(y1), txt))
    return cleaned


def _detect_two_column_blocks(blocks: list[tuple], page_width: float, page_height: float | None = None) -> tuple[bool, float]:
    if len(blocks) < 6:
        return False, 0.0
    usable = (
        _clean_blocks_for_layout(blocks, page_width, page_height if page_height is not None else page_width * 1.4)
        if page_height is not None
        else [(float(b[0]), float(b[1]), float(b[2]), float(b[3]), str(b[4])) for b in blocks if str(b[4]).strip()]
    )
    centers = []
    for block in usable:
        x0, _, x1, _, text = block[:5]
        if not str(text).strip():
            continue
        centers.append((float(x0) + float(x1)) / 2.0)
    if len(centers) < 8:
        return False, 0.0
    centers.sort()
    n = len(centers)
    left = centers[: n // 2]
    right = centers[n // 2 :]
    if not left or not right:
        return False, 0.0
    split = (left[-1] + right[0]) / 2.0
    separation = right[0] - left[-1]
    if separation < max(20.0, page_width * 0.08):
        return False, 0.0
    left_ratio = len(left) / n
    right_ratio = len(right) / n
    balanced = 0.25 <= left_ratio <= 0.75 and 0.25 <= right_ratio <= 0.75
    if not balanced:
        return False, 0.0
    # Require both columns to appear over multiple vertical bands (mixed layout guard).
    top = [b for b in usable if b[1] < (page_height or 1200) * 0.55]
    if top:
        l_top = sum(1 for b in top if ((b[0] + b[2]) / 2.0) <= split)
        r_top = sum(1 for b in top if ((b[0] + b[2]) / 2.0) > split)
        if l_top == 0 or r_top == 0:
            return False, 0.0
    return True, split


def _merge_region_counts(a: dict[str, int], b: dict[str, int]) -> dict[str, int]:
    out = dict(a)
    for k, v in b.items():
        out[k] = int(out.get(k, 0)) + int(v or 0)
    return out


def _extract_page_text_legacy(page, two_column_mode: str) -> tuple[str, bool]:
    blocks = page.get_text("blocks") or []
    page_w = float(page.rect.width)
    page_h = float(page.rect.height)
    cleaned_blocks = _clean_blocks_for_layout(blocks, page_w, page_h)
    if two_column_mode == "off":
        return "\n".join(b[4] for b in sorted(cleaned_blocks, key=lambda b: (b[1], b[0]))), False

    use_two_col = False
    split = 0.0
    if two_column_mode == "force":
        use_two_col = True
        split = page_w / 2.0
    elif two_column_mode == "auto":
        use_two_col, split = _detect_two_column_blocks(cleaned_blocks, page_w, page_h)

    if not use_two_col:
        return "\n".join(b[4] for b in sorted(cleaned_blocks, key=lambda b: (b[1], b[0]))), False

    left_blocks = []
    right_blocks = []
    for block in cleaned_blocks:
        x0, y0, x1, y1, text = block[:5]
        if not str(text).strip():
            continue
        xc = (float(x0) + float(x1)) / 2.0
        # Use y-bands then x-order to reduce interleave artifacts.
        band = int(float(y0) // 24)
        item = (band, float(y0), float(x0), str(text).strip())
        if xc <= split:
            left_blocks.append(item)
        else:
            right_blocks.append(item)

    left_blocks.sort(key=lambda t: (t[0], t[1], t[2]))
    right_blocks.sort(key=lambda t: (t[0], t[1], t[2]))
    merged = [t[3] for t in left_blocks] + [t[3] for t in right_blocks]
    return "\n".join(merged), True


def _extract_page_text_pymupdf(
    page,
    two_column_mode: str,
    layout_engine: str = "shadow",
    layout_table_handling: str = "linearize",
    layout_footnote_handling: str = "append",
    layout_min_region_confidence: float = 0.55,
) -> tuple[str, bool, dict[str, object]]:
    legacy_text, legacy_two_col = _extract_page_text_legacy(page, two_column_mode=two_column_mode)
    base_region_counts = {k: 0 for k in REGION_KEYS}
    stats: dict[str, object] = {
        "layout_v2_attempted": False,
        "layout_v2_applied": False,
        "layout_shadow_compared": False,
        "layout_shadow_diff": False,
        "layout_fallback_to_legacy": False,
        "layout_confidence": None,
        "layout_region_counts": base_region_counts,
        "layout_engine_used": "legacy",
    }
    if layout_engine not in {"legacy", "v2", "shadow"}:
        layout_engine = "shadow"
    if layout_engine == "legacy":
        return legacy_text, legacy_two_col, stats

    stats["layout_v2_attempted"] = True
    try:
        analysis = analyze_page_regions(page)
        v2 = reconstruct_page_text_v2(
            analysis,
            table_handling=layout_table_handling,
            footnote_handling=layout_footnote_handling,
        )
        stats["layout_region_counts"] = dict(v2.region_counts)
        stats["layout_confidence"] = round(float(v2.confidence), 4)
        v2_text = v2.text
    except Exception:
        stats["layout_fallback_to_legacy"] = True
        return legacy_text, legacy_two_col, stats

    if layout_engine == "shadow":
        legacy_norm = _compact_text(legacy_text)
        v2_norm = _compact_text(v2_text)
        shadow_diff = legacy_norm != v2_norm
        stats["layout_shadow_compared"] = True
        stats["layout_shadow_diff"] = shadow_diff
        stats["layout_engine_used"] = "legacy"
        return legacy_text, legacy_two_col, stats

    # layout_engine == v2
    if float(stats["layout_confidence"] or 0.0) < float(layout_min_region_confidence):
        stats["layout_fallback_to_legacy"] = True
        stats["layout_engine_used"] = "legacy"
        return legacy_text, legacy_two_col, stats

    stats["layout_v2_applied"] = True
    stats["layout_engine_used"] = "v2"
    return v2_text or legacy_text, legacy_two_col, stats


def _extract_text_pymupdf(
    pdf_path: str,
    two_column_mode: str = "auto",
    layout_engine: str = "shadow",
    layout_table_handling: str = "linearize",
    layout_footnote_handling: str = "append",
    layout_min_region_confidence: float = 0.55,
) -> tuple[str, list[dict[str, int | bool]], bool, dict[str, object]]:
    try:
        import fitz  # type: ignore
    except Exception as exc:
        raise ModuleNotFoundError("pymupdf not available") from exc
    doc = fitz.open(pdf_path)
    try:
        texts = []
        applied = False
        page_stats: list[dict[str, int | bool]] = []
        layout_region_counts = {k: 0 for k in REGION_KEYS}
        layout_v2_attempted_count = 0
        layout_v2_applied_count = 0
        layout_shadow_compared_count = 0
        layout_shadow_diff_count = 0
        layout_fallback_to_legacy_count = 0
        layout_conf_values: list[float] = []
        layout_engine_used = "legacy"
        for i, page in enumerate(doc, start=1):
            page_text, page_two_col, page_layout = _extract_page_text_pymupdf(
                page,
                two_column_mode=two_column_mode,
                layout_engine=layout_engine,
                layout_table_handling=layout_table_handling,
                layout_footnote_handling=layout_footnote_handling,
                layout_min_region_confidence=layout_min_region_confidence,
            )
            texts.append(page_text or "")
            applied = applied or page_two_col
            compact_len = len(_compact_text(page_text or ""))
            page_stats.append(
                {
                    "page": i,
                    "chars": compact_len,
                    "empty": compact_len == 0,
                    "layout_engine_used": page_layout.get("layout_engine_used"),
                    "layout_confidence": page_layout.get("layout_confidence"),
                    "layout_shadow_diff": bool(page_layout.get("layout_shadow_diff", False)),
                }
            )
            layout_v2_attempted_count += int(bool(page_layout.get("layout_v2_attempted", False)))
            layout_v2_applied_count += int(bool(page_layout.get("layout_v2_applied", False)))
            layout_shadow_compared_count += int(bool(page_layout.get("layout_shadow_compared", False)))
            layout_shadow_diff_count += int(bool(page_layout.get("layout_shadow_diff", False)))
            layout_fallback_to_legacy_count += int(bool(page_layout.get("layout_fallback_to_legacy", False)))
            if page_layout.get("layout_confidence") is not None:
                layout_conf_values.append(float(page_layout.get("layout_confidence")))
            page_counts = page_layout.get("layout_region_counts") or {}
            if isinstance(page_counts, dict):
                layout_region_counts = _merge_region_counts(layout_region_counts, {k: int(v or 0) for k, v in page_counts.items()})
            if page_layout.get("layout_engine_used") == "v2":
                layout_engine_used = "v2"
    finally:
        doc.close()
    compared = max(1, layout_shadow_compared_count)
    return "\n".join(texts), page_stats, applied, {
        "layout_engine_used": layout_engine_used if layout_engine != "shadow" else "legacy",
        "layout_v2_attempted_count": layout_v2_attempted_count,
        "layout_v2_applied_count": layout_v2_applied_count,
        "layout_shadow_compared_count": layout_shadow_compared_count,
        "layout_shadow_diff_count": layout_shadow_diff_count,
        "layout_shadow_diff_rate": round(layout_shadow_diff_count / compared, 4) if layout_shadow_compared_count > 0 else 0.0,
        "layout_region_counts": layout_region_counts,
        "layout_confidence_avg": round(sum(layout_conf_values) / len(layout_conf_values), 4) if layout_conf_values else None,
        "layout_fallback_to_legacy_count": layout_fallback_to_legacy_count,
    }


def _extract_text_pdfminer(pdf_path: str) -> tuple[str, list[dict[str, int | bool]]]:
    try:
        from pdfminer.high_level import extract_text  # type: ignore
    except Exception as exc:
        raise ModuleNotFoundError("pdfminer.six not available") from exc
    text = extract_text(pdf_path) or ""
    compact_len = len(_compact_text(text))
    return text, [{"page": 1, "chars": compact_len, "empty": compact_len == 0}]


def _classify_extract_exception(exc: Exception) -> str:
    msg = str(exc).lower()
    if "cryptography" in msg or "aes" in msg or "encrypted" in msg or "password" in msg:
        return "encrypted_or_unsupported"
    return "all_extractors_failed"


def _extract_text_tesseract_cli(
    pdf_path: str,
    timeout_sec: int = 30,
    max_pages: int = 20,
    lang: str = "eng",
    profile: str = "document",
) -> tuple[str, list[dict[str, int | bool]], float | None]:
    try:
        import fitz  # type: ignore
    except Exception as exc:
        raise RuntimeError("ocr_requires_pymupdf") from exc
    try:
        subprocess.run(["tesseract", "--version"], capture_output=True, text=True, check=True)
    except FileNotFoundError as exc:
        raise RuntimeError("ocr_binary_missing") from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError("ocr_failed") from exc

    doc = fitz.open(pdf_path)
    try:
        pages = min(len(doc), max_pages)
        texts: list[str] = []
        page_stats: list[dict[str, int | bool]] = []
        conf_values: list[float] = []
        psm = "6" if profile == "document" else "11"
        for i in range(pages):
            page = doc[i]
            pix = page.get_pixmap(dpi=220)
            with tempfile.TemporaryDirectory() as td:
                img_path = Path(td) / "page.png"
                out_base = Path(td) / "ocr_out"
                pix.save(str(img_path))
                try:
                    subprocess.run(
                        ["tesseract", str(img_path), str(out_base), "-l", lang, "--psm", psm],
                        capture_output=True,
                        text=True,
                        timeout=timeout_sec,
                        check=True,
                    )
                except subprocess.TimeoutExpired as exc:
                    raise RuntimeError("ocr_timeout") from exc
                except subprocess.CalledProcessError as exc:
                    raise RuntimeError("ocr_failed") from exc
                txt_path = out_base.with_suffix(".txt")
                page_text = txt_path.read_text(encoding="utf-8", errors="ignore") if txt_path.exists() else ""
                # Parse tsv confidence if available.
                tsv_path = out_base.with_suffix(".tsv")
                if tsv_path.exists():
                    try:
                        rows = csv.DictReader(tsv_path.read_text(encoding="utf-8", errors="ignore").splitlines(), delimiter="\t")
                        for row in rows:
                            conf = row.get("conf")
                            if conf is None:
                                continue
                            try:
                                v = float(conf)
                            except ValueError:
                                continue
                            if v >= 0:
                                conf_values.append(v)
                    except Exception:
                        pass
                compact_len = len(_compact_text(page_text))
                texts.append(page_text)
                page_stats.append({"page": i + 1, "chars": compact_len, "empty": compact_len == 0})
    finally:
        doc.close()
    avg_conf = (sum(conf_values) / len(conf_values)) if conf_values else None
    return "\n".join(texts), page_stats, avg_conf


def _suppress_ocr_noise(text: str) -> str:
    lines = []
    for raw in text.splitlines():
        line = re.sub(r"[^\x20-\x7E]", " ", raw)
        line = re.sub(r"\s+", " ", line).strip()
        if not line:
            continue
        # Drop lines that are mostly punctuation/symbol noise.
        alnum = sum(1 for c in line if c.isalnum())
        if alnum == 0:
            continue
        ratio = alnum / max(1, len(line))
        if ratio < 0.25:
            continue
        lines.append(line)
    return "\n".join(lines)


def _extract_text_with_fallback(
    pdf_path: str,
    two_column_mode: str = "auto",
    layout_engine: str = "shadow",
    layout_table_handling: str = "linearize",
    layout_footnote_handling: str = "append",
    layout_min_region_confidence: float = 0.55,
) -> tuple[str | None, str | None, str | None, bool, list[dict[str, int | bool]], dict[str, object]]:
    def _norm(result: object, default_two_col: bool = False) -> tuple[str, list[dict[str, int | bool]], bool, dict[str, object]]:
        if isinstance(result, tuple):
            if len(result) == 4:
                text, page_stats, two_col, layout_stats = result
                return str(text or ""), list(page_stats or []), bool(two_col), dict(layout_stats or {})
            if len(result) == 3:
                text, page_stats, two_col = result
                return str(text or ""), list(page_stats or []), bool(two_col), {}
            if len(result) == 2:
                text, page_stats = result
                return str(text or ""), list(page_stats or []), bool(default_two_col), {}
            if len(result) == 1:
                return str(result[0] or ""), [], bool(default_two_col), {}
        return str(result or ""), [], bool(default_two_col), {}

    attempts = [
        (
            "pymupdf",
            lambda p: _norm(
                _extract_text_pymupdf(
                    p,
                    two_column_mode=two_column_mode,
                    layout_engine=layout_engine,
                    layout_table_handling=layout_table_handling,
                    layout_footnote_handling=layout_footnote_handling,
                    layout_min_region_confidence=layout_min_region_confidence,
                )
            ),
        ),
        ("pdfminer", lambda p: _norm(_extract_text_pdfminer(p))),
        ("pypdf", lambda p: _norm(_extract_text_pypdf(p))),
    ]
    saw_missing_dep = False
    saw_encrypted_or_unsupported = False
    last_exc: Exception | None = None
    for name, fn in attempts:
        try:
            text, page_stats, two_col_applied, layout_stats = fn(pdf_path)
            return text, name, None, two_col_applied, page_stats, layout_stats
        except ModuleNotFoundError:
            saw_missing_dep = True
            continue
        except Exception as exc:
            if _classify_extract_exception(exc) == "encrypted_or_unsupported":
                saw_encrypted_or_unsupported = True
            last_exc = exc
            continue

    if saw_missing_dep and last_exc is None:
        return None, None, "extractor_missing_dependency", False, [], {}
    if saw_encrypted_or_unsupported:
        return None, None, "encrypted_or_unsupported", False, [], {}
    if last_exc is not None:
        return None, None, _classify_extract_exception(last_exc), False, [], {}
    return None, None, "all_extractors_failed", False, [], {}


def _normalize_heading(line: str) -> str:
    lowered = re.sub(r"^\d+(\.\d+)*\s*", "", line.strip().lower()).strip(":")
    return lowered


def _detect_section_from_heading(line: str) -> str | None:
    norm = _normalize_heading(line)
    for key, section in SECTION_HINTS.items():
        if norm == key or norm.startswith(key):
            return section
    return None


def _looks_like_heading(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    words = stripped.split()
    if len(words) > 10:
        return False
    has_terminal_punct = stripped.endswith(".") or stripped.endswith(",")
    return stripped == stripped.upper() or stripped.istitle() or not has_terminal_punct


def _split_paragraphs_with_sections(text: str) -> list[tuple[str, str]]:
    lines = [ln.rstrip() for ln in text.splitlines()]
    current_section = "body"
    paragraphs: list[tuple[str, str]] = []
    buf: list[str] = []

    def flush() -> None:
        nonlocal buf
        if buf:
            para = " ".join(x.strip() for x in buf if x.strip()).strip()
            if para:
                paragraphs.append((current_section, para))
            buf = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            flush()
            continue
        maybe_section = _detect_section_from_heading(stripped) if _looks_like_heading(stripped) else None
        if maybe_section:
            flush()
            current_section = maybe_section
            continue
        buf.append(stripped)
    flush()
    return paragraphs


def _chunk_text_with_sections(text: str, max_words: int = 120) -> list[tuple[str, str]]:
    paras = _split_paragraphs_with_sections(text)
    chunks: list[tuple[str, str]] = []
    for section, para in paras:
        words = para.split()
        if len(words) <= max_words:
            chunks.append((section, para))
            continue
        for i in range(0, len(words), max_words):
            chunks.append((section, " ".join(words[i : i + max_words])))
    return chunks


def compute_extraction_quality(text: str) -> tuple[float, str, dict[str, float | bool]]:
    compact = _compact_text(text)
    if not compact:
        return 0.0, "poor", {
            "compact_len": 0.0,
            "alnum_ratio": 0.0,
            "weird_ratio": 1.0,
            "avg_line_len": 0.0,
            "has_headings": False,
        }
    total = len(compact)
    alnum = sum(1 for c in compact if c.isalnum() or c.isspace())
    weird = sum(1 for c in compact if ord(c) < 9 or (14 <= ord(c) < 32) or ord(c) == 65533)
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    avg_line_len = (sum(len(ln) for ln in lines) / len(lines)) if lines else float(total)
    lower = compact.lower()
    has_headings = any(h in lower for h in ("abstract", "introduction", "references"))

    len_score = min(1.0, total / 3000.0)
    alnum_ratio = alnum / max(total, 1)
    weird_ratio = weird / max(total, 1)
    alnum_score = min(1.0, max(0.0, alnum_ratio))
    weird_score = max(0.0, 1.0 - min(1.0, weird_ratio * 6.0))
    line_len_score = 1.0 - min(1.0, abs(avg_line_len - 80.0) / 120.0)
    heading_score = 1.0 if has_headings else 0.65

    score = (
        0.35 * len_score
        + 0.20 * alnum_score
        + 0.20 * weird_score
        + 0.15 * line_len_score
        + 0.10 * heading_score
    )
    score = max(0.0, min(1.0, float(score)))
    band = "good" if score >= 0.75 else ("ok" if score >= 0.45 else "poor")
    return score, band, {
        "compact_len": float(total),
        "alnum_ratio": float(alnum_ratio),
        "weird_ratio": float(weird_ratio),
        "avg_line_len": float(avg_line_len),
        "has_headings": bool(has_headings),
    }


def extract_snippets(
    paper: PaperRecord,
    text: str,
    extraction_quality_score: float | None = None,
    extraction_quality_band: str | None = None,
    extraction_source: str | None = None,
) -> list[SnippetRecord]:
    chunks = _chunk_text_with_sections(text)
    out: list[SnippetRecord] = []
    for i, (section, chunk) in enumerate(chunks, start=1):
        out.append(
            SnippetRecord(
                snippet_id=snippet_id(paper.paper_id, i),
                paper_id=paper.paper_id,
                section=section,
                text=chunk,
                page_hint=None,
                token_count=len(chunk.split()),
                paper_year=paper.year,
                paper_venue=paper.venue,
                citation_count=paper.citation_count,
                extraction_quality_score=extraction_quality_score,
                extraction_quality_band=extraction_quality_band,
                extraction_source=extraction_source,
            )
        )
    return out


def save_extracted(
    paper: PaperRecord,
    snippets: list[SnippetRecord],
    out_dir: str,
    extraction_meta: dict[str, object] | None = None,
) -> str:
    out_path = Path(out_dir) / f"{paper.paper_id.replace('/', '_').replace(':', '_')}.json"
    payload = {
        "paper": paper.model_dump(),
        "snippets": [s.model_dump() for s in snippets],
    }
    if extraction_meta is not None:
        payload["extraction_meta"] = extraction_meta
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return str(out_path)


def load_snippets(corpus_dir: str) -> list[SnippetRecord]:
    snippets: list[SnippetRecord] = []
    for path in Path(corpus_dir).glob("*.json"):
        data = json.loads(path.read_text(encoding="utf-8"))
        paper = data.get("paper", {})
        for s in data.get("snippets", []):
            s.setdefault("paper_year", paper.get("year"))
            s.setdefault("paper_venue", paper.get("venue"))
            s.setdefault("citation_count", int(paper.get("citation_count") or 0))
            snippets.append(SnippetRecord(**s))
    return snippets


def _is_low_quality_text(text: str, min_text_chars: int) -> bool:
    return len(_compact_text(text)) < min_text_chars


def _record_failure(
    failed_pdfs: list[dict[str, str]],
    failed_reason_counts: Counter,
    pdf_path: Path,
    reason: str,
) -> None:
    failed_reason_counts[reason] += 1
    if len(failed_pdfs) < FAILED_PDFS_CAP:
        failed_pdfs.append({"path": str(pdf_path), "reason": reason})


def _build_quality_stats(scores: list[float], quality_band_counts: Counter) -> dict[str, object]:
    if not scores:
        return {"quality_score_avg": 0.0, "quality_score_min": 0.0, "quality_score_max": 0.0, "quality_band_counts": dict(quality_band_counts)}
    return {
        "quality_score_avg": round(sum(scores) / len(scores), 4),
        "quality_score_min": round(min(scores), 4),
        "quality_score_max": round(max(scores), 4),
        "quality_band_counts": dict(quality_band_counts),
    }


def _detect_ocr_language(native_text: str, fallback: str = "eng") -> tuple[str, bool]:
    text = native_text or ""
    if not text.strip():
        return fallback, False

    # Script heuristics for non-latin languages.
    cyr = sum(1 for c in text if "\u0400" <= c <= "\u04FF")
    ara = sum(1 for c in text if "\u0600" <= c <= "\u06FF")
    han = sum(1 for c in text if "\u4E00" <= c <= "\u9FFF")
    kana = sum(1 for c in text if "\u3040" <= c <= "\u30FF")
    hangul = sum(1 for c in text if "\uAC00" <= c <= "\uD7AF")
    if cyr >= 8:
        return "rus", True
    if ara >= 8:
        return "ara", True
    if kana >= 4:
        return "jpn", True
    if hangul >= 4:
        return "kor", True
    if han >= 8:
        return "chi_sim", True

    compact = _compact_text(text).lower()
    if len(compact) < 40:
        return fallback, False
    words = re.findall(r"[a-zA-ZÀ-ÿ]+", compact)
    if not words:
        return fallback, False
    vocab = set(words)
    scored: list[tuple[str, int]] = []
    for lang, stop in _OCR_STOPWORDS.items():
        score = sum(1 for w in stop if w in vocab)
        scored.append((lang, score))
    scored.sort(key=lambda x: x[1], reverse=True)
    best_lang, best_score = scored[0]
    second_score = scored[1][1] if len(scored) > 1 else 0
    if best_score >= 2 and best_score >= second_score:
        # Keep English mixed in latin-language OCR for technical papers.
        if best_lang != "eng":
            return f"eng+{best_lang}", True
        return "eng", True
    return fallback, False


def _extract_common(
    paper_paths: list[Path],
    paper_lookup: dict[str, PaperRecord] | None,
    out_dir: str,
    min_text_chars: int = 200,
    two_column_mode: str = "auto",
    ocr_enabled: bool = False,
    ocr_timeout_sec: int = 30,
    ocr_min_chars_trigger: int = 120,
    ocr_max_pages: int = 20,
    ocr_min_output_chars: int = 200,
    ocr_min_gain_chars: int = 40,
    ocr_min_confidence: float = 45.0,
    ocr_lang: str = "eng",
    ocr_profile: str = "document",
    ocr_noise_suppression: bool = True,
    layout_engine: str = "shadow",
    layout_table_handling: str = "linearize",
    layout_footnote_handling: str = "append",
    layout_min_region_confidence: float = 0.55,
) -> dict[str, object]:
    processed = 0
    created = 0
    resolved_from_db = 0
    extract_errors = 0
    empty_text_skipped = 0
    low_text_skipped = 0
    two_column_applied_count = 0
    ocr_attempted_count = 0
    ocr_succeeded_count = 0
    ocr_failed_count = 0
    ocr_rejected_low_quality_count = 0
    ocr_rejected_low_confidence_count = 0
    ocr_avg_confidence_values: list[float] = []
    ocr_lang_auto_detected_count = 0
    ocr_lang_detected_counts: Counter = Counter()
    extractor_used_counts: Counter = Counter()
    failed_reason_counts: Counter = Counter()
    failed_pdfs: list[dict[str, str]] = []
    quality_band_counts: Counter = Counter()
    quality_scores: list[float] = []
    layout_v2_attempted_count = 0
    layout_v2_applied_count = 0
    layout_shadow_compared_count = 0
    layout_shadow_diff_count = 0
    layout_fallback_to_legacy_count = 0
    layout_confidence_values: list[float] = []
    layout_region_counts: Counter = Counter()

    for pdf_path in paper_paths:
        processed += 1
        matched = paper_lookup.get(pdf_path.name) if paper_lookup is not None else None
        if matched is not None:
            paper = matched.model_copy(update={"pdf_path": str(pdf_path)})
            resolved_from_db += 1
        else:
            paper_id = pdf_path.stem.replace("_", ":")
            paper = PaperRecord(
                paper_id=paper_id,
                title=paper_id,
                authors=[],
                year=None,
                venue=None,
                source="local",
                abstract="",
                pdf_path=str(pdf_path),
                url="https://local.invalid",
                doi=None,
                arxiv_id=None,
                openalex_id=None,
                citation_count=0,
                is_open_access=False,
                sync_timestamp="2026-01-01T00:00:00Z",
            )

        extract_result = _extract_text_with_fallback(
            str(pdf_path),
            two_column_mode=two_column_mode,
            layout_engine=layout_engine,
            layout_table_handling=layout_table_handling,
            layout_footnote_handling=layout_footnote_handling,
            layout_min_region_confidence=layout_min_region_confidence,
        )
        # Backward compatibility for tests/monkeypatches that return legacy 4-tuple.
        if len(extract_result) == 4:
            text, extractor_name, failure_reason, two_col_applied = extract_result  # type: ignore[misc]
            page_stats: list[dict[str, int | bool]] = []
            layout_stats: dict[str, object] = {}
        else:
            if len(extract_result) == 5:
                text, extractor_name, failure_reason, two_col_applied, page_stats = extract_result  # type: ignore[misc]
                layout_stats = {}
            else:
                text, extractor_name, failure_reason, two_col_applied, page_stats, layout_stats = extract_result  # type: ignore[misc]
        if text is None:
            extract_errors += 1
            _record_failure(failed_pdfs, failed_reason_counts, pdf_path, failure_reason or "all_extractors_failed")
            continue
        if two_col_applied:
            two_column_applied_count += 1
        extractor_used_counts[extractor_name or "unknown"] += 1
        layout_v2_attempted_count += int(layout_stats.get("layout_v2_attempted_count", 0) or 0)
        layout_v2_applied_count += int(layout_stats.get("layout_v2_applied_count", 0) or 0)
        layout_shadow_compared_count += int(layout_stats.get("layout_shadow_compared_count", 0) or 0)
        layout_shadow_diff_count += int(layout_stats.get("layout_shadow_diff_count", 0) or 0)
        layout_fallback_to_legacy_count += int(layout_stats.get("layout_fallback_to_legacy_count", 0) or 0)
        if layout_stats.get("layout_confidence_avg") is not None:
            try:
                layout_confidence_values.append(float(layout_stats.get("layout_confidence_avg")))
            except Exception:
                pass
        if isinstance(layout_stats.get("layout_region_counts"), dict):
            for k, v in (layout_stats.get("layout_region_counts") or {}).items():  # type: ignore[assignment]
                layout_region_counts[str(k)] += int(v or 0)

        extraction_source = "native"
        compact_len = len(_compact_text(text))
        ocr_applied = False
        ocr_lang_used = ocr_lang
        if ocr_enabled and compact_len < ocr_min_chars_trigger:
            ocr_attempted_count += 1
            if str(ocr_lang).strip().lower() in {"auto", "auto+eng", "eng+auto"}:
                detected_lang, detected = _detect_ocr_language(text, fallback="eng")
                ocr_lang_used = detected_lang
                if detected:
                    ocr_lang_auto_detected_count += 1
                ocr_lang_detected_counts[ocr_lang_used] += 1
            try:
                try:
                    ocr_result = _extract_text_tesseract_cli(
                        str(pdf_path),
                        timeout_sec=ocr_timeout_sec,
                        max_pages=ocr_max_pages,
                        lang=ocr_lang_used,
                        profile=ocr_profile,
                    )
                except TypeError:
                    # Backward compatibility for monkeypatched or legacy call signatures in tests.
                    ocr_result = _extract_text_tesseract_cli(
                        str(pdf_path),
                        timeout_sec=ocr_timeout_sec,
                        max_pages=ocr_max_pages,
                    )
                ocr_avg_conf: float | None = None
                if isinstance(ocr_result, tuple):
                    if len(ocr_result) >= 3:
                        ocr_text = str(ocr_result[0] or "")
                        ocr_page_stats = list(ocr_result[1] or [])
                        try:
                            ocr_avg_conf = float(ocr_result[2]) if ocr_result[2] is not None else None
                        except Exception:
                            ocr_avg_conf = None
                    elif len(ocr_result) >= 2:
                        ocr_text = str(ocr_result[0] or "")
                        ocr_page_stats = list(ocr_result[1] or [])
                    else:
                        ocr_text = str(ocr_result[0] or "")
                        ocr_page_stats = []
                else:
                    ocr_text = str(ocr_result or "")
                    ocr_page_stats = []
                if ocr_noise_suppression:
                    ocr_text = _suppress_ocr_noise(ocr_text)
                ocr_compact_len = len(_compact_text(ocr_text))
                min_required = max(int(ocr_min_output_chars), int(compact_len + ocr_min_gain_chars))
                if ocr_avg_conf is not None and ocr_avg_conf < float(ocr_min_confidence):
                    ocr_failed_count += 1
                    ocr_rejected_low_confidence_count += 1
                    _record_failure(failed_pdfs, failed_reason_counts, pdf_path, "ocr_low_confidence")
                elif ocr_compact_len >= min_required:
                    text = ocr_text
                    page_stats = ocr_page_stats
                    extraction_source = "ocr"
                    ocr_applied = True
                    ocr_succeeded_count += 1
                    if ocr_avg_conf is not None:
                        ocr_avg_confidence_values.append(ocr_avg_conf)
                else:
                    ocr_failed_count += 1
                    ocr_rejected_low_quality_count += 1
                    _record_failure(failed_pdfs, failed_reason_counts, pdf_path, "ocr_low_quality")
            except RuntimeError as exc:
                reason = str(exc)
                if reason not in {"ocr_binary_missing", "ocr_timeout", "ocr_failed", "ocr_requires_pymupdf"}:
                    reason = "ocr_failed"
                ocr_failed_count += 1
                _record_failure(failed_pdfs, failed_reason_counts, pdf_path, reason)

        if not text.strip():
            empty_text_skipped += 1
            _record_failure(failed_pdfs, failed_reason_counts, pdf_path, "empty_text")
            continue
        if _is_low_quality_text(text, min_text_chars):
            low_text_skipped += 1
            _record_failure(failed_pdfs, failed_reason_counts, pdf_path, "low_text")
            continue

        quality_score, quality_band, _signals = compute_extraction_quality(text)
        quality_scores.append(quality_score)
        quality_band_counts[quality_band] += 1

        snippets = extract_snippets(
            paper,
            text,
            extraction_quality_score=quality_score,
            extraction_quality_band=quality_band,
            extraction_source=extraction_source,
        )
        pages_total = len(page_stats)
        empty_pages = sum(1 for ps in page_stats if bool(ps.get("empty", False)))
        save_extracted(
            paper,
            snippets,
            out_dir,
            extraction_meta={
                "extractor": extractor_name,
                "two_column_applied": bool(two_col_applied),
                "ocr_applied": bool(ocr_applied),
                "layout_engine_used": layout_stats.get("layout_engine_used", "legacy"),
                "layout_confidence": layout_stats.get("layout_confidence_avg"),
                "layout_region_counts": layout_stats.get("layout_region_counts", {}),
                "layout_shadow_diff": bool((layout_stats.get("layout_shadow_diff_count", 0) or 0) > 0),
                "ocr_lang_used": ocr_lang_used,
                "pages_total": pages_total,
                "empty_pages": empty_pages,
                "empty_page_pct": round((empty_pages / pages_total), 4) if pages_total > 0 else 0.0,
                "page_stats": page_stats,
                "quality_score": round(quality_score, 4),
                "quality_band": quality_band,
            },
        )
        created += 1

    stats: dict[str, object] = {
        "processed": processed,
        "created": created,
        "extract_errors": extract_errors,
        "empty_text_skipped": empty_text_skipped,
        "low_text_skipped": low_text_skipped,
        "min_text_chars": min_text_chars,
        "failed_pdfs_count": int(sum(failed_reason_counts.values())),
        "failed_reason_counts": dict(failed_reason_counts),
        "extractor_used_counts": dict(extractor_used_counts),
        "failed_pdfs": failed_pdfs,
        "two_column_mode": two_column_mode,
        "two_column_applied_count": two_column_applied_count,
        "layout_engine": layout_engine,
        "layout_table_handling": layout_table_handling,
        "layout_footnote_handling": layout_footnote_handling,
        "layout_min_region_confidence": layout_min_region_confidence,
        "layout_v2_attempted_count": layout_v2_attempted_count,
        "layout_v2_applied_count": layout_v2_applied_count,
        "layout_shadow_compared_count": layout_shadow_compared_count,
        "layout_shadow_diff_rate": round(layout_shadow_diff_count / layout_shadow_compared_count, 4)
        if layout_shadow_compared_count > 0
        else 0.0,
        "layout_region_counts": dict(layout_region_counts),
        "layout_confidence_avg": round(sum(layout_confidence_values) / len(layout_confidence_values), 4)
        if layout_confidence_values
        else None,
        "layout_fallback_to_legacy_count": layout_fallback_to_legacy_count,
        "ocr_enabled": ocr_enabled,
        "ocr_attempted_count": ocr_attempted_count,
        "ocr_succeeded_count": ocr_succeeded_count,
        "ocr_failed_count": ocr_failed_count,
        "ocr_rejected_low_quality_count": ocr_rejected_low_quality_count,
        "ocr_rejected_low_confidence_count": ocr_rejected_low_confidence_count,
        "ocr_avg_confidence": round(sum(ocr_avg_confidence_values) / len(ocr_avg_confidence_values), 3)
        if ocr_avg_confidence_values
        else None,
        "ocr_lang_auto_detected_count": ocr_lang_auto_detected_count,
        "ocr_lang_detected_counts": dict(ocr_lang_detected_counts),
        "ocr_min_chars_trigger": ocr_min_chars_trigger,
        "ocr_timeout_sec": ocr_timeout_sec,
        "ocr_max_pages": ocr_max_pages,
        "ocr_min_output_chars": ocr_min_output_chars,
        "ocr_min_gain_chars": ocr_min_gain_chars,
        "ocr_min_confidence": ocr_min_confidence,
        "ocr_lang": ocr_lang,
        "ocr_profile": ocr_profile,
        "ocr_noise_suppression": ocr_noise_suppression,
    }
    stats.update(_build_quality_stats(quality_scores, quality_band_counts))
    if paper_lookup is not None:
        stats["resolved_from_db"] = resolved_from_db
    return stats


def extract_from_papers_dir(
    papers_dir: str,
    out_dir: str,
    min_text_chars: int = 200,
    two_column_mode: str = "auto",
    ocr_enabled: bool = False,
    ocr_timeout_sec: int = 30,
    ocr_min_chars_trigger: int = 120,
    ocr_max_pages: int = 20,
    ocr_min_output_chars: int = 200,
    ocr_min_gain_chars: int = 40,
    ocr_min_confidence: float = 45.0,
    ocr_lang: str = "eng",
    ocr_profile: str = "document",
    ocr_noise_suppression: bool = True,
    layout_engine: str = "shadow",
    layout_table_handling: str = "linearize",
    layout_footnote_handling: str = "append",
    layout_min_region_confidence: float = 0.55,
) -> dict[str, object]:
    return _extract_common(
        paper_paths=list(Path(papers_dir).glob("*.pdf")),
        paper_lookup=None,
        out_dir=out_dir,
        min_text_chars=min_text_chars,
        two_column_mode=two_column_mode,
        ocr_enabled=ocr_enabled,
        ocr_timeout_sec=ocr_timeout_sec,
        ocr_min_chars_trigger=ocr_min_chars_trigger,
        ocr_max_pages=ocr_max_pages,
        ocr_min_output_chars=ocr_min_output_chars,
        ocr_min_gain_chars=ocr_min_gain_chars,
        ocr_min_confidence=ocr_min_confidence,
        ocr_lang=ocr_lang,
        ocr_profile=ocr_profile,
        ocr_noise_suppression=ocr_noise_suppression,
        layout_engine=layout_engine,
        layout_table_handling=layout_table_handling,
        layout_footnote_handling=layout_footnote_handling,
        layout_min_region_confidence=layout_min_region_confidence,
    )


def _paper_filename(paper_id: str) -> str:
    return paper_id.replace(":", "_").replace("/", "_") + ".pdf"


def _load_papers_by_filename(db_path: str) -> dict[str, PaperRecord]:
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            "SELECT paper_id, title, authors, year, venue, source, abstract, pdf_path, url, doi, arxiv_id, openalex_id, citation_count, is_open_access, sync_timestamp FROM papers"
        ).fetchall()
    out: dict[str, PaperRecord] = {}
    for r in rows:
        p = PaperRecord(
            paper_id=r[0],
            title=r[1],
            authors=r[2].split("|") if r[2] else [],
            year=r[3],
            venue=r[4],
            source=r[5],
            abstract=r[6] or "",
            pdf_path=r[7],
            url=r[8],
            doi=r[9],
            arxiv_id=r[10],
            openalex_id=r[11],
            citation_count=int(r[12] or 0),
            is_open_access=bool(r[13]),
            sync_timestamp=r[14],
        )
        out[_paper_filename(p.paper_id)] = p
    return out


def extract_from_papers_dir_with_db(
    papers_dir: str,
    out_dir: str,
    db_path: str,
    min_text_chars: int = 200,
    two_column_mode: str = "auto",
    ocr_enabled: bool = False,
    ocr_timeout_sec: int = 30,
    ocr_min_chars_trigger: int = 120,
    ocr_max_pages: int = 20,
    ocr_min_output_chars: int = 200,
    ocr_min_gain_chars: int = 40,
    ocr_min_confidence: float = 45.0,
    ocr_lang: str = "eng",
    ocr_profile: str = "document",
    ocr_noise_suppression: bool = True,
    layout_engine: str = "shadow",
    layout_table_handling: str = "linearize",
    layout_footnote_handling: str = "append",
    layout_min_region_confidence: float = 0.55,
) -> dict[str, object]:
    filename_map = _load_papers_by_filename(db_path)
    return _extract_common(
        paper_paths=list(Path(papers_dir).glob("*.pdf")),
        paper_lookup=filename_map,
        out_dir=out_dir,
        min_text_chars=min_text_chars,
        two_column_mode=two_column_mode,
        ocr_enabled=ocr_enabled,
        ocr_timeout_sec=ocr_timeout_sec,
        ocr_min_chars_trigger=ocr_min_chars_trigger,
        ocr_max_pages=ocr_max_pages,
        ocr_min_output_chars=ocr_min_output_chars,
        ocr_min_gain_chars=ocr_min_gain_chars,
        ocr_min_confidence=ocr_min_confidence,
        ocr_lang=ocr_lang,
        ocr_profile=ocr_profile,
        ocr_noise_suppression=ocr_noise_suppression,
        layout_engine=layout_engine,
        layout_table_handling=layout_table_handling,
        layout_footnote_handling=layout_footnote_handling,
        layout_min_region_confidence=layout_min_region_confidence,
    )
