from __future__ import annotations

from pathlib import Path
from typing import Any
import json
import re

import yaml

from src.apps.direct_answer import try_direct_answer


_CITATION_NOTICE = "ASK mode does not provide evidence citations; use RESEARCH mode for grounded citations."


def _load_glossary(path: str | None) -> dict[str, str]:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        return {}
    block = data.get("glossary", data)
    if not isinstance(block, dict):
        return {}
    out: dict[str, str] = {}
    for k, v in block.items():
        kk = str(k).strip().lower()
        vv = str(v).strip()
        if kk and vv:
            out[kk] = vv
    return out


def _format_num(x: float) -> str:
    if abs(x - round(x)) < 1e-9:
        return str(int(round(x)))
    return (f"{x:.8f}").rstrip("0").rstrip(".")


def _convert_units(question: str) -> tuple[bool, str]:
    q = question.strip().lower()
    m = re.search(r"([-+]?\d+(?:\.\d+)?)\s*(km/?h|kph)\s*(?:to|in|as)\s*(m/?s|mps)", q)
    if m:
        v = float(m.group(1)) / 3.6
        return True, f"{_format_num(v)} m/s"

    m = re.search(r"([-+]?\d+(?:\.\d+)?)\s*(m/?s|mps)\s*(?:to|in|as)\s*(km/?h|kph)", q)
    if m:
        v = float(m.group(1)) * 3.6
        return True, f"{_format_num(v)} km/h"

    m = re.search(r"([-+]?\d+(?:\.\d+)?)\s*%\s*(?:to|in|as)?\s*decimal", q)
    if m:
        v = float(m.group(1)) / 100.0
        return True, _format_num(v)

    m = re.search(r"([-+]?\d+(?:\.\d+)?)\s*(?:decimal)\s*(?:to|in|as)\s*percent", q)
    if m:
        v = float(m.group(1)) * 100.0
        return True, f"{_format_num(v)}%"

    return False, ""


_DEF_TRAILERS = [
    "in simple terms",
    "simply",
    "briefly",
    "for beginners",
    "for a beginner",
    "please",
]


def _clean_definition_term(term: str) -> str:
    t = (term or "").strip().strip(" ?.!\"'`")
    t = re.sub(r"^(a|an|the)\s+", "", t, flags=re.IGNORECASE)
    low = t.lower()
    for suffix in _DEF_TRAILERS:
        if low.endswith(" " + suffix):
            t = t[: -len(suffix)].strip(" ,;:-")
            low = t.lower()
    return re.sub(r"\s+", " ", t).strip()


def _extract_definition_terms(question: str, glossary: dict[str, str]) -> list[str]:
    q = question.strip().lower()
    candidates: list[str] = []

    patterns = [
        r"^\s*(?:what is|what are|define|explain)\s+(.+?)\s*$",
        r".*?\b(?:what is|what are|define|explain)\s+(.+?)(?:\?|$)",
    ]
    for pat in patterns:
        m = re.match(pat, q, flags=re.IGNORECASE)
        if m:
            cand = _clean_definition_term(m.group(1))
            if cand:
                candidates.append(cand)

    for key in glossary.keys():
        if re.search(rf"\b{re.escape(key)}\b", q):
            candidates.append(key)

    out: list[str] = []
    seen: set[str] = set()
    for c in candidates:
        if c and c not in seen:
            seen.add(c)
            out.append(c)
    return out


def answer_ask_question(
    *,
    question: str,
    router_cfg: dict[str, Any],
    glossary_path: str | None = "config/ask_glossary.yaml",
) -> tuple[str, dict[str, Any]]:
    ask_cfg = ((router_cfg.get("router", {}) or {}).get("ask", {}) if isinstance(router_cfg, dict) else {}) or {}
    citation_kw = [str(x).strip().lower() for x in (ask_cfg.get("citation_notice_on_keywords", []) or []) if str(x).strip()]
    q_lower = question.lower()
    citation_notice = bool(any(k in q_lower for k in citation_kw))

    direct = try_direct_answer(question, max_complexity=3)
    if direct.used:
        answer = str(direct.value)
        md = f"# Ask\n\n{answer}\n"
        if citation_notice:
            md += f"\n> {_CITATION_NOTICE}\n"
        meta = {
            "mode": "ask",
            "ask_handler_used": "direct_arithmetic",
            "ask_fallback_used": False,
            "deterministic": True,
            "no_citation_notice_emitted": citation_notice,
        }
        return md, meta

    converted, conv = _convert_units(question)
    if converted:
        md = f"# Ask\n\n{conv}\n"
        if citation_notice:
            md += f"\n> {_CITATION_NOTICE}\n"
        meta = {
            "mode": "ask",
            "ask_handler_used": "unit_conversion",
            "ask_fallback_used": False,
            "deterministic": True,
            "no_citation_notice_emitted": citation_notice,
        }
        return md, meta

    glossary = _load_glossary(glossary_path)
    def_terms = _extract_definition_terms(question, glossary)
    for term in def_terms:
        term_l = term.lower()
        glossary_hit = glossary.get(term_l)
        if not glossary_hit and term_l.endswith("s"):
            glossary_hit = glossary.get(term_l[:-1])
        if not glossary_hit and not term_l.endswith("s"):
            glossary_hit = glossary.get(term_l + "s")
        if glossary_hit:
            md = f"# Ask\n\n{glossary_hit}\n"
            if citation_notice:
                md += f"\n> {_CITATION_NOTICE}\n"
            meta = {
                "mode": "ask",
                "ask_handler_used": "glossary",
                "ask_fallback_used": False,
                "deterministic": True,
                "no_citation_notice_emitted": citation_notice,
            }
            return md, meta

    if def_terms:
        term_preview = def_terms[0]
        fallback = (
            f"I don't have a reliable local ASK definition for \"{term_preview}\". "
            "Use RESEARCH mode for grounded output."
        )
        md = f"# Ask\n\n{fallback}\n"
        if citation_notice:
            md += f"\n> {_CITATION_NOTICE}\n"
        meta = {
            "mode": "ask",
            "ask_handler_used": "definition_fallback",
            "ask_fallback_used": True,
            "deterministic": True,
            "no_citation_notice_emitted": citation_notice,
        }
        return md, meta

    fallback = "I can't provide a reliable quick answer in ASK mode for this prompt. Use RESEARCH for grounded output."
    md = f"# Ask\n\n{fallback}\n"
    if citation_notice:
        md += f"\n> {_CITATION_NOTICE}\n"
    meta = {
        "mode": "ask",
        "ask_handler_used": "fallback",
        "ask_fallback_used": True,
        "deterministic": True,
        "no_citation_notice_emitted": citation_notice,
    }
    return md, meta


def run_ask_mode(
    *,
    question: str,
    out_path: str,
    router_cfg: dict[str, Any],
    glossary_path: str | None = "config/ask_glossary.yaml",
) -> dict[str, Any]:
    md, meta = answer_ask_question(question=question, router_cfg=router_cfg, glossary_path=glossary_path)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md, encoding="utf-8")
    sidecar = out.with_suffix(".json")
    sidecar.write_text(
        json.dumps(
            {
                "question": question,
                "mode": "ask",
                **meta,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return {"out_path": str(out), "sidecar_path": str(sidecar), **meta}
