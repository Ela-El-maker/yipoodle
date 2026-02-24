from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import re

_NEGATION_TOKENS = {"no", "not", "never", "without", "none", "cannot", "can't", "dont", "don't"}


def canonicalize_claim(text: str) -> str:
    s = (text or "").strip().lower()
    s = re.sub(r"\b(\d+)\.0\b", r"\1", s)
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def canonical_hash(text: str) -> str:
    return sha256(canonicalize_claim(text).encode("utf-8")).hexdigest()


def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(v)))


def compute_claim_confidence(
    *,
    mean_support: float,
    citation_quality: float,
    recency: float,
    validation_signal: float,
) -> float:
    score = (
        0.35 * _clamp(mean_support)
        + 0.25 * _clamp(citation_quality)
        + 0.20 * _clamp(recency)
        + 0.20 * _clamp(validation_signal)
    )
    return round(_clamp(score), 4)


def apply_confidence_decay(confidence: float, *, days_since_confirmed: float, decay_per_day: float = 0.98) -> float:
    days = max(0.0, float(days_since_confirmed))
    d = max(0.0, min(1.0, float(decay_per_day)))
    return round(_clamp(float(confidence) * (d ** days)), 4)


def parse_iso_utc(ts: str | None) -> datetime | None:
    if not ts:
        return None
    s = str(ts).strip()
    if not s:
        return None
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(s)
    except Exception:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def contradiction_score(a: str, b: str) -> float:
    ta = set(re.findall(r"[a-z0-9]+", canonicalize_claim(a)))
    tb = set(re.findall(r"[a-z0-9]+", canonicalize_claim(b)))
    if not ta or not tb:
        return 0.0
    overlap = len(ta & tb) / max(1, len(ta | tb))
    a_neg = any(t in _NEGATION_TOKENS for t in ta)
    b_neg = any(t in _NEGATION_TOKENS for t in tb)
    neg_mismatch = 1.0 if a_neg != b_neg else 0.0
    if overlap < 0.3:
        return 0.0
    return round(_clamp(0.65 * neg_mismatch + 0.35 * overlap), 4)
