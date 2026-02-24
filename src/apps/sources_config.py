from __future__ import annotations

from pathlib import Path
from typing import Any
import os
import re

import yaml


SUPPORTED_SYNC_SOURCES = {
    "arxiv",
    "openalex",
    "semanticscholar",
    "unpaywall",
    "crossref",
    "dblp",
    "paperswithcode",
    "core",
    "openreview",
    "github",
    "zenodo",
    "opencitations",
    "springer",
    "ieee_xplore",
    "figshare",
    "openml",
    "gdelt",
    "wikidata",
    "orcid",
}
_ENV_REF = re.compile(r"^\$\{([A-Za-z_][A-Za-z0-9_]*)\}$")


def load_sources_config(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    try:
        payload = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def source_enabled(cfg: dict[str, Any], source: str) -> bool:
    sources = cfg.get("sources", {}) if isinstance(cfg, dict) else {}
    block = sources.get(source, {}) if isinstance(sources, dict) else {}
    if isinstance(block, dict) and "enabled" in block:
        return bool(block.get("enabled"))
    return True


def source_max_results(cfg: dict[str, Any], source: str, default_max: int) -> int:
    sources = cfg.get("sources", {}) if isinstance(cfg, dict) else {}
    block = sources.get(source, {}) if isinstance(sources, dict) else {}
    configured = None
    if isinstance(block, dict):
        configured = block.get("max_results")
    try:
        c = int(configured) if configured is not None else int(default_max)
        return max(1, min(int(default_max), c))
    except Exception:
        return int(default_max)


def source_endpoint(cfg: dict[str, Any], source: str, default: str | None = None) -> str | None:
    sources = cfg.get("sources", {}) if isinstance(cfg, dict) else {}
    block = sources.get(source, {}) if isinstance(sources, dict) else {}
    if isinstance(block, dict) and block.get("endpoint"):
        return str(block.get("endpoint"))
    return default


def source_required_param(cfg: dict[str, Any], source: str, key: str) -> str | None:
    sources = cfg.get("sources", {}) if isinstance(cfg, dict) else {}
    block = sources.get(source, {}) if isinstance(sources, dict) else {}
    req = block.get("required_query_params", {}) if isinstance(block, dict) else {}
    if not isinstance(req, dict):
        return None
    raw = req.get(key)
    if raw is None:
        return None
    val = str(raw).strip()
    m = _ENV_REF.match(val)
    if m:
        return os.getenv(m.group(1))
    return val


def source_auth_header(cfg: dict[str, Any], source: str) -> dict[str, str] | None:
    sources = cfg.get("sources", {}) if isinstance(cfg, dict) else {}
    block = sources.get(source, {}) if isinstance(sources, dict) else {}
    auth = block.get("auth", {}) if isinstance(block, dict) else {}
    if not isinstance(auth, dict):
        return None
    header = auth.get("header")
    value = auth.get("value")
    if not header or value is None:
        return None
    h = str(header).strip()
    v = str(value).strip()
    if not h or not v:
        return None
    m = _ENV_REF.match(v)
    if m:
        v = os.getenv(m.group(1), "").strip()
    if not v:
        return None
    return {h: v}


def source_auth_query_param(cfg: dict[str, Any], source: str) -> dict[str, str] | None:
    sources = cfg.get("sources", {}) if isinstance(cfg, dict) else {}
    block = sources.get(source, {}) if isinstance(sources, dict) else {}
    auth = block.get("auth", {}) if isinstance(block, dict) else {}
    if not isinstance(auth, dict):
        return None
    key = auth.get("query_param")
    value = auth.get("value")
    if not key or value is None:
        return None
    k = str(key).strip()
    v = str(value).strip()
    if not k or not v:
        return None
    m = _ENV_REF.match(v)
    if m:
        v = os.getenv(m.group(1), "").strip()
    if not v:
        return None
    return {k: v}


def max_tokens_per_summary(cfg: dict[str, Any]) -> int | None:
    limits = cfg.get("limits", {}) if isinstance(cfg, dict) else {}
    value = limits.get("max_tokens_per_summary")
    if value is None:
        return None
    try:
        v = int(value)
        return v if v > 0 else None
    except Exception:
        return None


def ocr_config(cfg: dict[str, Any]) -> dict[str, Any]:
    block = cfg.get("ocr", {}) if isinstance(cfg, dict) else {}
    if not isinstance(block, dict):
        return {}

    out: dict[str, Any] = {}

    if "enabled" in block:
        out["enabled"] = bool(block.get("enabled"))
    if "noise_suppression" in block:
        out["noise_suppression"] = bool(block.get("noise_suppression"))

    int_keys = {
        "timeout_sec": 1,
        "min_chars_trigger": 1,
        "max_pages": 1,
        "min_output_chars": 1,
        "min_gain_chars": 0,
    }
    for k, min_v in int_keys.items():
        if k not in block:
            continue
        try:
            out[k] = max(min_v, int(block.get(k)))
        except Exception:
            continue

    if "min_confidence" in block:
        try:
            out["min_confidence"] = max(0.0, min(100.0, float(block.get("min_confidence"))))
        except Exception:
            pass

    if "lang" in block:
        lang = str(block.get("lang") or "").strip()
        if lang:
            out["lang"] = lang

    if "profile" in block:
        profile = str(block.get("profile") or "").strip().lower()
        if profile in {"document", "sparse"}:
            out["profile"] = profile

    return out


def metadata_prior_weight(cfg: dict[str, Any], default: float = 0.2) -> float:
    ranking = cfg.get("ranking", {}) if isinstance(cfg, dict) else {}
    weights = ranking.get("weights", {}) if isinstance(ranking, dict) else {}
    # Map configured ranking weights into a single metadata prior strength.
    # Keep bounded to avoid overpowering lexical relevance.
    recency = float(weights.get("recency", 0.0) or 0.0)
    citation = float(weights.get("citation_count", 0.0) or 0.0)
    source_trust = float(weights.get("source_trust", 0.0) or 0.0)
    total = recency + citation + source_trust
    if total <= 0:
        return float(default)
    return max(0.0, min(0.5, total * 0.5))


def unsupported_enabled_sources(cfg: dict[str, Any]) -> list[str]:
    sources = cfg.get("sources", {}) if isinstance(cfg, dict) else {}
    if not isinstance(sources, dict):
        return []
    out: list[str] = []
    for name, block in sources.items():
        if not isinstance(block, dict):
            continue
        if bool(block.get("enabled", False)) and name not in SUPPORTED_SYNC_SOURCES:
            out.append(str(name))
    return sorted(out)
