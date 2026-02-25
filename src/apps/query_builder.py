from __future__ import annotations

from dataclasses import dataclass
import re

TOKEN_RE = re.compile(r"[a-zA-Z0-9]+")

CV_EXPANSIONS: dict[str, list[str]] = {
    "segmentation": ["matting", "mask", "foreground", "background", "portrait"],
    "matting": ["segmentation", "alpha", "trimap", "boundary"],
    "mobile": ["realtime", "real-time", "lightweight", "efficient", "edge"],
    "real-time": ["realtime", "fps", "latency", "throughput"],
    "realtime": ["real-time", "fps", "latency", "throughput"],
    "video": ["temporal", "frame", "stability"],
    "boundary": ["edge", "hair", "fine", "detail"],
    "low": ["night", "dark", "illumination"],
}

DEFAULT_SECTION_WEIGHTS = {
    "abstract": 1.15,
    "introduction": 1.0,
    "related_work": 0.95,
    "method": 1.2,
    "experiments": 1.15,
    "results": 1.25,
    "limitations": 1.35,
    "future_work": 1.25,
    "discussion": 1.1,
    "conclusion": 1.0,
    "body": 1.0,
}


@dataclass
class QueryPlan:
    query_terms: list[str]
    term_boosts: dict[str, float]
    section_weights: dict[str, float]


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in TOKEN_RE.findall(text)]


def build_query_plan(question: str, domain: str = "computer_vision") -> QueryPlan:
    base_tokens = tokenize(question)
    term_boosts: dict[str, float] = {}

    for tok in base_tokens:
        term_boosts[tok] = max(term_boosts.get(tok, 0.0), 1.0)

    if domain == "computer_vision":
        for tok in list(base_tokens):
            for exp in CV_EXPANSIONS.get(tok, []):
                exp_norm = exp.replace("-", "")
                term_boosts[exp_norm] = max(term_boosts.get(exp_norm, 0.0), 0.75)

    query_terms = sorted(term_boosts.keys())
    return QueryPlan(
        query_terms=query_terms,
        term_boosts=term_boosts,
        section_weights=DEFAULT_SECTION_WEIGHTS.copy(),
    )
