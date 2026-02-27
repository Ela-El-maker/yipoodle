from __future__ import annotations

from pathlib import Path
from typing import Any
import json
import os
import re

import requests
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
        r"^\s*(?:what is|what are|who is|who are|who was|define|explain)\s+(.+?)\s*$",
        r".*?\b(?:what is|what are|who is|who are|who was|define|explain)\s+(.+?)(?:\?|$)",
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


def _ask_model_fallback(question: str, ask_cfg: dict[str, Any]) -> tuple[str | None, str | None]:
    mf = ask_cfg.get("model_fallback", {}) if isinstance(ask_cfg.get("model_fallback"), dict) else {}
    if not bool(mf.get("enabled", False)):
        return None, None

    base_url = (
        mf.get("base_url")
        or os.getenv("ASK_BASE_URL")
        or os.getenv("DEEPSEEK_BASE_URL")
        or os.getenv("OPENAI_BASE_URL")
        or "https://api.openai.com/v1"
    )
    api_key = (
        mf.get("api_key")
        or os.getenv("ASK_API_KEY")
        or os.getenv("DEEPSEEK_API_KEY")
        or os.getenv("OPENAI_API_KEY")
    )
    model = (
        mf.get("model")
        or os.getenv("ASK_MODEL")
        or os.getenv("DEEPSEEK_MODEL")
        or os.getenv("OPENAI_MODEL")
        or "gpt-4o-mini"
    )
    timeout_sec = float(mf.get("timeout_sec", 15))
    max_tokens = int(mf.get("max_tokens", 256))
    temperature = float(mf.get("temperature", 0.2))
    if not api_key:
        return None, "ask_model_fallback_missing_api_key"

    system_prompt = str(
        mf.get("system_prompt")
        or "You are a concise assistant. Answer directly in 2-5 sentences. Do not invent citations."
    )
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    try:
        resp = requests.post(
            str(base_url).rstrip("/") + "/chat/completions",
            headers=headers,
            json=payload,
            timeout=timeout_sec,
        )
        resp.raise_for_status()
        data = resp.json()
        content = (
            (((data.get("choices") or [{}])[0].get("message") or {}).get("content"))
            if isinstance(data, dict)
            else None
        )
        text = str(content or "").strip()
        if not text:
            return None, "ask_model_fallback_empty_response"
        return text, None
    except Exception as exc:
        return None, f"ask_model_fallback_error:{exc}"


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

    model_answer, model_err = _ask_model_fallback(question, ask_cfg)
    if model_answer:
        md = f"# Ask\n\n{model_answer}\n"
        if citation_notice:
            md += f"\n> {_CITATION_NOTICE}\n"
        meta = {
            "mode": "ask",
            "ask_handler_used": "model_fallback",
            "ask_fallback_used": True,
            "deterministic": False,
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
            "model_fallback_error": model_err,
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
        "model_fallback_error": model_err,
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
