from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import re


REGION_KEYS = ("body_left", "body_right", "table", "caption", "footnote", "header", "footer", "other")


@dataclass
class LayoutRegion:
    x0: float
    y0: float
    x1: float
    y1: float
    text: str
    label: str
    band: int


@dataclass
class LayoutPageAnalysis:
    regions: list[LayoutRegion]
    split_x: float
    page_width: float
    page_height: float
    region_counts: dict[str, int]


@dataclass
class LayoutPageResult:
    text: str
    confidence: float
    region_counts: dict[str, int]
    used_v2: bool
    used_fallback: bool
    shadow_diff: bool


def _compact(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _is_caption(text: str) -> bool:
    t = text.lower().strip()
    return t.startswith(("fig.", "figure ", "table ", "tab.", "algorithm "))


def _looks_table(text: str) -> bool:
    t = _compact(text)
    if not t:
        return False
    # common separators in extracted table rows
    if "|" in t or "\t" in text:
        return True
    if re.search(r"\S\s{2,}\S", text):
        return True
    numeric = sum(1 for c in t if c.isdigit())
    punct = sum(1 for c in t if c in {":", ";", ",", "-", "_", "%"})
    return len(t) > 25 and numeric >= 6 and punct >= 4


def _is_footnote(text: str, y0: float, page_h: float) -> bool:
    t = text.strip()
    low = t.lower()
    if page_h > 0 and y0 > page_h * 0.84:
        if re.match(r"^\d+[\].)]\s", t):
            return True
        if re.match(r"^[*†‡]\s", t):
            return True
        if low.startswith(("note:", "notes:", "footnote", "*")):
            return True
        if len(t) < 220 and re.search(r"\bdoi\b|\bhttp\b|\barxiv\b", low):
            return True
    return False


def _region_label(text: str, x0: float, x1: float, y0: float, y1: float, split_x: float, page_h: float) -> str:
    t = text.strip()
    if not t:
        return "other"
    if page_h > 0 and y1 < page_h * 0.06 and len(t) < 140:
        return "header"
    if page_h > 0 and y0 > page_h * 0.95 and len(t) < 140:
        return "footer"
    if _is_caption(t):
        return "caption"
    if _is_footnote(t, y0, page_h):
        return "footnote"
    if _looks_table(t):
        return "table"
    xc = (x0 + x1) / 2.0
    return "body_left" if xc <= split_x else "body_right"


def analyze_page_regions(page: Any) -> LayoutPageAnalysis:
    blocks = page.get_text("blocks") or []
    page_w = float(page.rect.width)
    page_h = float(page.rect.height)
    split_x = page_w / 2.0

    regions: list[LayoutRegion] = []
    counts = {k: 0 for k in REGION_KEYS}
    for block in blocks:
        x0, y0, x1, y1, text = block[:5]
        txt = str(text or "").strip()
        if not txt:
            continue
        label = _region_label(txt, float(x0), float(x1), float(y0), float(y1), split_x, page_h)
        band = int(float(y0) // 24)
        reg = LayoutRegion(float(x0), float(y0), float(x1), float(y1), txt, label, band)
        regions.append(reg)
        counts[label] = counts.get(label, 0) + 1
    return LayoutPageAnalysis(
        regions=regions,
        split_x=split_x,
        page_width=page_w,
        page_height=page_h,
        region_counts=counts,
    )


def layout_confidence(analysis: LayoutPageAnalysis, out_text: str) -> float:
    total = max(1, len(analysis.regions))
    body = analysis.region_counts.get("body_left", 0) + analysis.region_counts.get("body_right", 0)
    table = analysis.region_counts.get("table", 0)
    foot = analysis.region_counts.get("footnote", 0)
    cap = analysis.region_counts.get("caption", 0)
    header_footer = analysis.region_counts.get("header", 0) + analysis.region_counts.get("footer", 0)

    body_ratio = body / total
    aux_ratio = (table + foot + cap) / total
    clean_ratio = max(0.0, 1.0 - (header_footer / total))
    text_len_score = min(1.0, len(_compact(out_text)) / 2200.0)
    # Penalize ambiguous wide body blocks that cross the column split;
    # these are common in mixed-layout pages (tables/figures embedded in flow).
    wide_cross = 0
    for r in analysis.regions:
        if r.label not in {"body_left", "body_right"}:
            continue
        if r.x0 < analysis.split_x * 0.7 and r.x1 > analysis.split_x * 1.3:
            wide_cross += 1
    wide_penalty = min(0.4, wide_cross / max(1, total))

    score = 0.45 * body_ratio + 0.20 * clean_ratio + 0.20 * text_len_score + 0.15 * (1.0 - min(0.9, aux_ratio))
    score *= 1.0 - wide_penalty
    return max(0.0, min(1.0, float(score)))


def reconstruct_page_text_v2(
    analysis: LayoutPageAnalysis,
    table_handling: str = "linearize",
    footnote_handling: str = "append",
) -> LayoutPageResult:
    if table_handling not in {"drop", "linearize", "preserve"}:
        table_handling = "linearize"
    if footnote_handling not in {"drop", "append", "preserve"}:
        footnote_handling = "append"

    by_label: dict[str, list[LayoutRegion]] = {k: [] for k in REGION_KEYS}
    for r in analysis.regions:
        by_label.setdefault(r.label, []).append(r)

    def sort_regs(rows: list[LayoutRegion]) -> list[LayoutRegion]:
        return sorted(rows, key=lambda t: (t.band, t.y0, t.x0))

    parts: list[str] = []
    # Keep short full-width headings before body streams.
    for r in sort_regs(by_label.get("other", [])):
        if len(_compact(r.text)) <= 120:
            parts.append(r.text)
    # main body: left then right
    for r in sort_regs(by_label.get("body_left", [])):
        parts.append(r.text)
    for r in sort_regs(by_label.get("body_right", [])):
        parts.append(r.text)

    if table_handling == "linearize":
        for r in sort_regs(by_label.get("table", [])):
            parts.append(_compact(r.text))
    elif table_handling == "preserve":
        for r in sort_regs(by_label.get("table", [])):
            parts.append(r.text)

    # captions kept after body/tables to reduce interruption
    for r in sort_regs(by_label.get("caption", [])):
        parts.append(r.text)

    if footnote_handling in {"append", "preserve"}:
        for r in sort_regs(by_label.get("footnote", [])):
            parts.append(r.text if footnote_handling == "preserve" else _compact(r.text))

    out_text = "\n".join(p for p in parts if p.strip())
    conf = layout_confidence(analysis, out_text)
    return LayoutPageResult(
        text=out_text,
        confidence=conf,
        region_counts=analysis.region_counts,
        used_v2=True,
        used_fallback=False,
        shadow_diff=False,
    )
