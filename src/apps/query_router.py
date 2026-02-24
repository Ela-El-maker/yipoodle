from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import re

import yaml

from src.apps.direct_answer import try_direct_answer


@dataclass(frozen=True)
class RouterDecision:
    mode: str
    confidence: float
    reason: str
    signals: dict[str, Any]
    override_used: bool


@dataclass(frozen=True)
class RouterContext:
    question: str
    explicit_mode: str | None = None


_DEFAULT_CFG: dict[str, Any] = {
    "router": {
        "default_mode": "auto",
        "ask": {
            "max_words_short_question": 14,
            "citation_notice_on_keywords": ["cite", "citation", "literature", "sources", "papers"],
            "definition_patterns": ["what is", "define", "explain"],
            "model_fallback": {"enabled": False},
        },
        "monitor": {
            "intent_keywords": ["monitor", "track", "notify", "alert"],
            "default_schedule_cron": "0 */6 * * *",
        },
        "notes": {
            "intent_keywords": ["notes", "study notes", "summarize as notes", "store this"],
        },
    }
}


def load_router_config(path: str | None) -> dict[str, Any]:
    cfg = dict(_DEFAULT_CFG)
    if not path:
        return cfg
    p = Path(path)
    if not p.exists():
        return cfg
    data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        return cfg
    router = data.get("router")
    if isinstance(router, dict):
        out_router = dict(cfg["router"])
        for k, v in router.items():
            if isinstance(v, dict) and isinstance(out_router.get(k), dict):
                merged = dict(out_router[k])
                merged.update(v)
                out_router[k] = merged
            else:
                out_router[k] = v
        cfg["router"] = out_router
    return cfg


def _contains_any(text: str, phrases: list[str]) -> list[str]:
    hits: list[str] = []
    for p in phrases:
        q = str(p).strip().lower()
        if not q:
            continue
        if " " in q:
            if q in text:
                hits.append(q)
        else:
            if re.search(rf"\b{re.escape(q)}\b", text):
                hits.append(q)
    return sorted(set(hits))


def route_query(question: str, cfg: dict[str, Any], explicit_mode: str | None = None) -> RouterDecision:
    q = (question or "").strip()
    t = q.lower()
    router = cfg.get("router", {}) if isinstance(cfg, dict) else {}
    ask_cfg = router.get("ask", {}) if isinstance(router.get("ask"), dict) else {}
    mon_cfg = router.get("monitor", {}) if isinstance(router.get("monitor"), dict) else {}
    notes_cfg = router.get("notes", {}) if isinstance(router.get("notes"), dict) else {}

    if explicit_mode and explicit_mode != "auto":
        mode = str(explicit_mode).strip().lower()
        if mode in {"ask", "research", "monitor", "notes"}:
            return RouterDecision(
                mode=mode,
                confidence=1.0,
                reason="explicit_mode_override",
                signals={"explicit_mode": mode},
                override_used=True,
            )

    mon_hits = _contains_any(t, list(mon_cfg.get("intent_keywords", [])))
    if mon_hits:
        return RouterDecision(
            mode="monitor",
            confidence=min(1.0, 0.6 + 0.1 * len(mon_hits)),
            reason="monitor_keywords",
            signals={"monitor_hits": mon_hits},
            override_used=False,
        )

    notes_hits = _contains_any(t, list(notes_cfg.get("intent_keywords", [])))
    if notes_hits:
        return RouterDecision(
            mode="notes",
            confidence=min(1.0, 0.6 + 0.1 * len(notes_hits)),
            reason="notes_keywords",
            signals={"notes_hits": notes_hits},
            override_used=False,
        )

    if try_direct_answer(q, max_complexity=3).used:
        return RouterDecision(
            mode="ask",
            confidence=0.95,
            reason="direct_arithmetic_candidate",
            signals={"math_candidate": True},
            override_used=False,
        )

    has_unit = bool(
        re.search(r"\b(km/?h|kph|m/?s|mps|percent|%)\b", t)
        and re.search(r"\b(to|in|as)\b", t)
        and re.search(r"\d", t)
    )
    def_pats = [str(x).strip().lower() for x in (ask_cfg.get("definition_patterns", []) or []) if str(x).strip()]
    def_hits = [p for p in def_pats if t.startswith(p + " ") or t == p]
    def_anywhere = bool(re.search(r"\b(what is|what are|define|explain)\b", t))
    short_cap = int(ask_cfg.get("max_words_short_question", 14) or 14)
    short = len(q.split()) <= short_cap
    if has_unit or ((def_hits or def_anywhere) and short):
        return RouterDecision(
            mode="ask",
            confidence=0.8 if def_hits else 0.75,
            reason="ask_pattern_match",
            signals={
                "unit_candidate": has_unit,
                "definition_hits": def_hits,
                "definition_anywhere": def_anywhere,
                "short": short,
                "short_cap": short_cap,
            },
            override_used=False,
        )

    return RouterDecision(
        mode="research",
        confidence=0.7,
        reason="default_research_fallback",
        signals={},
        override_used=False,
    )


def dispatch_query(decision: RouterDecision, handlers: dict[str, Any]) -> Any:
    fn = handlers.get(decision.mode)
    if fn is None:
        raise ValueError(f"No handler for mode: {decision.mode}")
    return fn()
