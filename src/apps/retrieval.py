from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import json
import math
import re

from src.core.schemas import EvidenceItem, EvidencePack, SnippetRecord

TOKEN_RE = re.compile(r"[a-zA-Z0-9]+")


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in TOKEN_RE.findall(text)]


def _metadata_prior(snippet: SnippetRecord, current_year: int | None = None) -> float:
    if current_year is None:
        from datetime import datetime
        current_year = datetime.now().year
    # Mild multiplicative prior: lexical BM25 remains dominant.
    year_prior = 1.0
    if snippet.paper_year is not None:
        age = max(0, current_year - snippet.paper_year)
        year_prior = max(0.85, 1.10 - min(age, 12) * 0.015)

    venue_prior = 1.0
    if snippet.paper_venue:
        venue = snippet.paper_venue.lower()
        if any(v in venue for v in ("cvpr", "iccv", "eccv", "neurips", "icml", "tpami")):
            venue_prior = 1.08

    citation_prior = 1.0 + min(0.20, math.log1p(max(0, snippet.citation_count)) / 30.0)
    return year_prior * venue_prior * citation_prior


def _quality_prior(snippet: SnippetRecord, quality_prior_weight: float = 0.15) -> float:
    q = snippet.extraction_quality_score if snippet.extraction_quality_score is not None else 1.0
    q = min(1.0, max(0.0, float(q)))
    return 1.0 - quality_prior_weight * (1.0 - q)


@dataclass
class SimpleBM25Index:
    snippets: list[SnippetRecord]
    idf: dict[str, float]
    doc_tf: list[Counter]
    avg_len: float

    @classmethod
    def build(cls, snippets: list[SnippetRecord]) -> "SimpleBM25Index":
        doc_tf: list[Counter] = []
        df: Counter = Counter()
        lengths = []
        for s in snippets:
            toks = tokenize(s.text)
            tf = Counter(toks)
            doc_tf.append(tf)
            lengths.append(len(toks))
            for t in tf.keys():
                df[t] += 1

        n_docs = len(snippets)
        idf = {t: math.log(1 + (n_docs - c + 0.5) / (c + 0.5)) for t, c in df.items()}
        avg_len = (sum(lengths) / n_docs) if n_docs else 1.0
        return cls(snippets=snippets, idf=idf, doc_tf=doc_tf, avg_len=avg_len)

    def query(
        self,
        question: str,
        top_k: int = 8,
        k1: float = 1.5,
        b: float = 0.75,
        query_terms: list[str] | None = None,
        term_boosts: dict[str, float] | None = None,
        section_weights: dict[str, float] | None = None,
        max_per_paper: int = 2,
        use_metadata_priors: bool = True,
        metadata_prior_weight: float = 0.2,
        quality_prior_weight: float = 0.15,
    ) -> EvidencePack:
        ranked = self.query_scored(
            question=question,
            top_k=top_k,
            k1=k1,
            b=b,
            query_terms=query_terms,
            term_boosts=term_boosts,
            section_weights=section_weights,
            max_per_paper=max_per_paper,
            use_metadata_priors=use_metadata_priors,
            metadata_prior_weight=metadata_prior_weight,
            quality_prior_weight=quality_prior_weight,
        )
        return evidence_from_ranked(question, ranked, top_k=top_k, max_per_paper=max_per_paper)

    def query_scored(
        self,
        question: str,
        top_k: int = 8,
        k1: float = 1.5,
        b: float = 0.75,
        query_terms: list[str] | None = None,
        term_boosts: dict[str, float] | None = None,
        section_weights: dict[str, float] | None = None,
        max_per_paper: int | None = None,
        use_metadata_priors: bool = True,
        metadata_prior_weight: float = 0.2,
        quality_prior_weight: float = 0.15,
    ) -> list[tuple[SnippetRecord, float]]:
        q_toks = query_terms if query_terms is not None else tokenize(question)
        term_boosts = term_boosts or {}
        section_weights = section_weights or {}
        scores: list[tuple[int, float]] = []
        for i, sn in enumerate(self.snippets):
            tf = self.doc_tf[i]
            dl = max(sum(tf.values()), 1)
            score = 0.0
            for tok in q_toks:
                if tok not in tf:
                    continue
                denom = tf[tok] + k1 * (1 - b + b * dl / self.avg_len)
                boost = term_boosts.get(tok, 1.0)
                score += boost * self.idf.get(tok, 0.0) * (tf[tok] * (k1 + 1) / denom)
            if score > 0:
                score *= section_weights.get(sn.section, 1.0)
            if score > 0 and use_metadata_priors:
                prior = _metadata_prior(sn)
                # Blend rather than replace: final = lexical * (1 + w * (prior-1)).
                score *= 1.0 + metadata_prior_weight * (prior - 1.0)
            if score > 0:
                score *= _quality_prior(sn, quality_prior_weight=quality_prior_weight)
            if score > 0:
                scores.append((i, score))
        scores.sort(key=lambda x: x[1], reverse=True)

        ranked: list[tuple[SnippetRecord, float]] = []
        per_paper_count: Counter = Counter()
        for idx, score in scores:
            sn = self.snippets[idx]
            if max_per_paper is not None and per_paper_count[sn.paper_id] >= max_per_paper:
                continue
            ranked.append((sn, score))
            per_paper_count[sn.paper_id] += 1
            if len(ranked) >= top_k:
                break
        return ranked


def save_index(index: SimpleBM25Index, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "snippets": [s.model_dump() for s in index.snippets],
    }
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


@lru_cache(maxsize=16)
def _load_index_cached(path: str, mtime_ns: int) -> SimpleBM25Index:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    snippets = [SnippetRecord(**s) for s in payload["snippets"]]
    return SimpleBM25Index.build(snippets)


def load_index(path: str) -> SimpleBM25Index:
    p = Path(path)
    return _load_index_cached(str(p.resolve()), p.stat().st_mtime_ns)


def clear_index_cache() -> None:
    _load_index_cached.cache_clear()


def index_cache_info() -> dict[str, int]:
    info = _load_index_cached.cache_info()
    return {
        "hits": info.hits,
        "misses": info.misses,
        "maxsize": info.maxsize or 0,
        "currsize": info.currsize,
    }


def evidence_from_ranked(
    question: str,
    ranked: list[tuple[SnippetRecord, float]],
    top_k: int,
    max_per_paper: int = 2,
) -> EvidencePack:
    items: list[EvidenceItem] = []
    per_paper_count: Counter = Counter()
    for sn, score in ranked:
        if per_paper_count[sn.paper_id] >= max_per_paper:
            continue
        items.append(
            EvidenceItem(
                paper_id=sn.paper_id,
                snippet_id=sn.snippet_id,
                score=float(score),
                section=sn.section,
                text=sn.text,
                paper_year=sn.paper_year,
                paper_venue=sn.paper_venue,
                citation_count=sn.citation_count,
                extraction_quality_score=sn.extraction_quality_score,
                extraction_quality_band=sn.extraction_quality_band,
                extraction_source=sn.extraction_source,
            )
        )
        per_paper_count[sn.paper_id] += 1
        if len(items) >= top_k:
            break
    return EvidencePack(question=question, items=items)


def minmax_normalize_scores(scores_by_id: dict[str, float]) -> dict[str, float]:
    if not scores_by_id:
        return {}
    vals = list(scores_by_id.values())
    lo = min(vals)
    hi = max(vals)
    if hi == lo:
        return {k: 1.0 for k in scores_by_id}
    span = hi - lo
    return {k: (v - lo) / span for k, v in scores_by_id.items()}


def fuse_hybrid_scores(
    lexical_scores: dict[str, float],
    vector_scores: dict[str, float],
    alpha: float = 0.6,
) -> dict[str, float]:
    l_norm = minmax_normalize_scores(lexical_scores)
    v_norm = minmax_normalize_scores(vector_scores)
    merged_ids = set(l_norm.keys()) | set(v_norm.keys())
    return {sid: alpha * l_norm.get(sid, 0.0) + (1.0 - alpha) * v_norm.get(sid, 0.0) for sid in merged_ids}


def derive_vector_paths(
    lexical_index_path: str,
    vector_index_path: str | None = None,
    vector_metadata_path: str | None = None,
) -> tuple[str, str]:
    base = Path(lexical_index_path)
    idx = Path(vector_index_path) if vector_index_path else base.with_suffix(".faiss")
    meta = Path(vector_metadata_path) if vector_metadata_path else base.with_suffix(".vector_meta.json")
    return str(idx), str(meta)
