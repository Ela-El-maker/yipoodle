from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import re


@dataclass(frozen=True)
class IntentResult:
    intent: str
    confidence: float
    matched_terms: list[str]


_DEFAULT_INTENTS: dict[str, dict[str, Any]] = {
    "finance_price": {
        "match_terms": ["price", "stock", "ticker", "market", "usd", "crypto", "bitcoin", "ethereum"],
        "sources": ["coingecko_price", "yahoo_finance"],
    },
    "weather_now": {
        "match_terms": ["weather", "temperature", "rain", "forecast", "humidity"],
        "sources": ["open_meteo"],
    },
    "sports_results": {
        "match_terms": ["won", "score", "final", "fixture", "champions league", "premier league"],
        "sources": ["sports_results_api"],
    },
    "tech_news": {
        "match_terms": ["headline", "news", "startup", "ai", "gpu", "infrastructure"],
        "sources": ["hn_rss"],
    },
}


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def _live_routing_block(cfg: dict[str, Any]) -> dict[str, Any]:
    block = cfg.get("live_routing", {}) if isinstance(cfg, dict) else {}
    if not isinstance(block, dict):
        return {}
    return block


def has_live_routing(cfg: dict[str, Any]) -> bool:
    block = _live_routing_block(cfg)
    intents = block.get("intents")
    return isinstance(intents, dict) and bool(intents)


def _intents_cfg(cfg: dict[str, Any]) -> dict[str, dict[str, Any]]:
    block = _live_routing_block(cfg)
    intents = block.get("intents") if isinstance(block.get("intents"), dict) else None
    if not intents:
        return {}
    out: dict[str, dict[str, Any]] = {}
    for name, row in intents.items():
        if not isinstance(row, dict):
            continue
        terms = [str(x).strip().lower() for x in (row.get("match_terms") or []) if str(x).strip()]
        sources = [str(x).strip() for x in (row.get("sources") or []) if str(x).strip()]
        if not terms:
            continue
        out[str(name).strip()] = {"match_terms": terms, "sources": sources}
    return out


def default_routing_mode(cfg: dict[str, Any]) -> str:
    block = _live_routing_block(cfg)
    mode = str(block.get("default_mode") or "auto").strip().lower()
    return mode if mode in {"auto", "manual"} else "auto"


def detect_intent(question: str, cfg: dict[str, Any], forced_intent: str | None = None) -> IntentResult:
    if forced_intent:
        return IntentResult(intent=str(forced_intent).strip(), confidence=1.0, matched_terms=[])

    text = _normalize_text(question)
    intents = _intents_cfg(cfg)

    best_intent = "general"
    best_terms: list[str] = []
    best_score = 0
    for intent, row in intents.items():
        matched: list[str] = []
        for term in row.get("match_terms", []):
            t = str(term).strip().lower()
            if not t:
                continue
            if " " in t:
                if t in text:
                    matched.append(t)
            else:
                if re.search(rf"\b{re.escape(t)}\b", text):
                    matched.append(t)
        score = len(set(matched))
        if score > best_score:
            best_score = score
            best_intent = intent
            best_terms = sorted(set(matched))

    confidence = 0.0 if best_score <= 0 else min(1.0, 0.4 + 0.2 * best_score)
    return IntentResult(intent=best_intent, confidence=round(confidence, 4), matched_terms=best_terms)


def sources_for_intent(intent: str, cfg: dict[str, Any], enabled_live_sources: dict[str, Any]) -> list[str]:
    intents = _intents_cfg(cfg)
    row = intents.get(intent)
    if not row:
        return []
    configured = [str(x).strip() for x in (row.get("sources") or []) if str(x).strip()]
    return [x for x in configured if x in enabled_live_sources]


def routing_help_for_intent(intent: str, cfg: dict[str, Any]) -> list[str]:
    block = cfg.get("live_routing_help", {}) if isinstance(cfg, dict) else {}
    if not isinstance(block, dict):
        return []
    row = block.get(intent, {}) if isinstance(block.get(intent), dict) else {}
    vals = row.get("suggest_enable") if isinstance(row.get("suggest_enable"), list) else []
    return [str(x).strip() for x in vals if str(x).strip()]


def build_not_found_diagnostics(
    *,
    question: str,
    intent_result: IntentResult,
    attempted_sources: list[str],
    reject_reason: str,
    cfg: dict[str, Any],
) -> dict[str, Any]:
    suggestions = routing_help_for_intent(intent_result.intent, cfg)
    return {
        "question": question,
        "intent_detected": intent_result.intent,
        "intent_confidence": float(intent_result.confidence),
        "matched_terms": list(intent_result.matched_terms),
        "attempted_sources": list(attempted_sources),
        "reject_reason": reject_reason,
        "suggest_enable": suggestions,
    }
