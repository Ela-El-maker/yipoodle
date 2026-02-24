from __future__ import annotations

from hashlib import sha256
from pathlib import Path
import json
import re
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from typing import Any

from src.apps.live_snapshot_store import load_cached_snapshot, save_snapshot, snapshot_to_snippets
from src.apps.live_sources import fetch_live_source, live_snapshot_config, live_sources_config
from src.apps.live_routing import (
    IntentResult,
    build_not_found_diagnostics,
    default_routing_mode,
    detect_intent,
    has_live_routing,
    sources_for_intent,
)
from src.apps.direct_answer import try_direct_answer
from src.apps.kb_query import query_kb_as_evidence
from src.apps.query_builder import build_query_plan
from src.apps.retrieval import (
    SimpleBM25Index,
    derive_vector_paths,
    evidence_from_ranked,
    fuse_hybrid_scores,
    index_cache_info,
    load_index,
    minmax_normalize_scores,
)
from src.apps.vector_index import DEFAULT_EMBEDDING_MODEL, load_vector_index, query_vector_index
from src.apps.sources_config import load_sources_config, max_tokens_per_summary, metadata_prior_weight
from src.core.schemas import EvidenceItem, EvidencePack, ExperimentProposal, ResearchReport, ShortlistItem
from src.core.validation import (
    report_coverage_metrics,
    validate_claim_support,
    validate_no_new_numbers,
    validate_report_citations,
    validate_semantic_claim_support,
)


_Q_RELEVANCE_STOPWORDS = {
    # Question words
    "what", "which", "when", "where", "who", "whom", "whose",
    "why", "how",

    # Auxiliary / modal verbs
    "is", "are", "was", "were", "be", "been", "being",
    "do", "does", "did",
    "can", "could", "should", "would", "may", "might", "must",
    "shall", "will",

    # Articles
    "the", "a", "an",

    # Conjunctions
    "and", "or", "but", "nor", "yet", "so",

    # Prepositions
    "to", "of", "in", "on", "at", "by", "for", "from", "with",
    "about", "against", "between", "into", "through",
    "during", "before", "after", "above", "below",
    "under", "over", "within", "without", "across",
    "behind", "beyond", "around", "near",

    # Determiners
    "this", "that", "these", "those",
    "some", "any", "each", "every",
    "either", "neither",
    "much", "many", "few", "several",
    "all", "both", "such",

    # Pronouns
    "i", "me", "my", "mine",
    "you", "your", "yours",
    "he", "him", "his",
    "she", "her", "hers",
    "it", "its",
    "we", "us", "our", "ours",
    "they", "them", "their", "theirs",

    # Common verbs (low semantic value in search)
    "have", "has", "had",
    "make", "makes", "made",
    "get", "gets", "got",
    "know", "knows", "knew",
    "see", "sees", "saw",
    "want", "wants",
    "need", "needs",
    "use", "uses", "used",

    # Query filler words
    "tell", "explain", "describe",
    "show", "give", "provide",
    "find", "search", "look",
    "help", "guide",

    # Time fillers
    "today", "latest", "recent", "current",
    "now", "currently", "recently",
    "update", "updated",

    # General low-signal words
    "example", "examples",
    "information", "details",
    "difference", "differences",
    "overview", "introduction",
    "basics", "guide",
}


def _question_tokens(text: str) -> set[str]:
    toks = {t.lower() for t in re.findall(r"[A-Za-z0-9]+", text) if len(t) >= 3}
    return {t for t in toks if t not in _Q_RELEVANCE_STOPWORDS}


def _question_evidence_relevance(question: str, evidence: EvidencePack) -> dict[str, float | int]:
    q = _question_tokens(question)
    if not q or not evidence.items:
        return {
            "question_relevance_avg": 0.0,
            "question_relevance_max": 0.0,
            "question_relevance_min": 0.0,
            "question_shared_terms_max": 0,
        }
    scores: list[float] = []
    max_shared = 0
    for it in evidence.items:
        et = _question_tokens(it.text)
        shared = len(q & et)
        max_shared = max(max_shared, shared)
        scores.append(shared / max(1, len(q)))
    return {
        "question_relevance_avg": round(sum(scores) / len(scores), 4),
        "question_relevance_max": round(max(scores), 4),
        "question_relevance_min": round(min(scores), 4),
        "question_shared_terms_max": int(max_shared),
    }


def _snippet_quality_factor(sn, quality_prior_weight: float) -> float:
    q = sn.extraction_quality_score if getattr(sn, "extraction_quality_score", None) is not None else 1.0
    q = min(1.0, max(0.0, float(q)))
    return 1.0 - float(quality_prior_weight) * (1.0 - q)


def _apply_quality_to_vector_scores(
    score_map: dict[str, float],
    snippet_by_id: dict[str, object],
    quality_prior_weight: float,
) -> dict[str, float]:
    out: dict[str, float] = {}
    for sid, score in score_map.items():
        sn = snippet_by_id.get(sid)
        if sn is None:
            continue
        out[sid] = float(score) * _snippet_quality_factor(sn, quality_prior_weight=quality_prior_weight)
    return out


def _build_shortlist(evidence: EvidencePack) -> list[ShortlistItem]:
    best: dict[str, tuple[float, str]] = {}
    for item in evidence.items:
        cur = best.get(item.paper_id)
        meta: list[str] = []
        if item.paper_year is not None:
            meta.append(f"year={item.paper_year}")
        if item.paper_venue:
            meta.append(f"venue={item.paper_venue}")
        if item.citation_count:
            meta.append(f"citations={item.citation_count}")
        meta_blob = f"; {', '.join(meta)}" if meta else ""
        reason = (
            f"Lexical+metadata score={item.score:.3f} in {item.section} ({item.snippet_id})"
            f"{meta_blob}"
        )
        if cur is None or item.score > cur[0]:
            best[item.paper_id] = (item.score, reason)
    ranked = sorted(best.items(), key=lambda kv: kv[1][0], reverse=True)[:5]
    return [ShortlistItem(paper_id=pid, title=pid, reason=meta[1]) for pid, meta in ranked]


def _synthesize(evidence: EvidencePack) -> tuple[str, list[str], list[ExperimentProposal], list[str]]:
    if not evidence.items:
        return "Not found in sources.", ["Not found in sources."], [], []

    top = evidence.items[: min(6, len(evidence.items))]
    lines: list[str] = []
    gaps: list[str] = []
    exps: list[ExperimentProposal] = []
    citations: list[str] = []

    for item in top:
        cit = f"({item.snippet_id})"
        citations.append(cit)
        snippet_short = " ".join(item.text.split()[:24])
        lines.append(f"Claim: {snippet_short} ... {cit}")
        lower = item.text.lower()
        if "limitation" in lower or "future work" in lower or "fails" in lower:
            gaps.append(f"Potential gap around {item.section} {cit}")
            exps.append(ExperimentProposal(proposal=f"Test an ablation around {item.section} features {cit}", citations=[cit]))

    if not gaps:
        first_cit = citations[0]
        gaps.append(f"No explicit limitations extracted; gather more diverse evidence {first_cit}")
        exps.append(ExperimentProposal(proposal=f"Run baseline + one robustness ablation {first_cit}", citations=[first_cit]))

    return "\n".join(lines), gaps, exps, sorted(set(citations))


def _derive_key_claims(synthesis: str) -> list[str]:
    out: list[str] = []
    for ln in [x.strip() for x in synthesis.splitlines() if x.strip()]:
        lower = ln.lower()
        if "not found in sources." in lower or "insufficient evidence confidence" in lower:
            continue
        if "(" not in ln or ":S" not in ln:
            continue
        text = ln
        if text.lower().startswith("claim:"):
            text = text[6:].strip()
        text = re.sub(r"\s+", " ", text).strip(" .;:-\t")
        if text:
            out.append(text)
    dedup: list[str] = []
    seen: set[str] = set()
    for x in out:
        if x not in seen:
            dedup.append(x)
            seen.add(x)
    return dedup


def _build_insufficient_evidence_report(question: str, evidence: EvidencePack) -> ResearchReport:
    if not evidence.items:
        return ResearchReport(
            question=question,
            shortlist=[],
            synthesis="Not found in sources.",
            key_claims=[],
            gaps=["Insufficient evidence: retrieval returned no supporting snippets."],
            experiments=[],
            citations=[],
        )

    top = evidence.items[0]
    cit = f"({top.snippet_id})"
    return ResearchReport(
        question=question,
        shortlist=_build_shortlist(evidence),
        synthesis=f"Insufficient evidence confidence for strong claims {cit}",
        key_claims=[],
        gaps=[f"Need broader corpus/query expansion before drawing conclusions {cit}"],
        experiments=[],
        citations=[cit],
    )


def build_research_report(question: str, evidence: EvidencePack, min_items: int = 2, min_score: float = 0.5) -> ResearchReport:
    if len(evidence.items) < min_items or (evidence.items and evidence.items[0].score < min_score):
        report = _build_insufficient_evidence_report(question, evidence)
    else:
        report = ResearchReport(
            question=question,
            shortlist=_build_shortlist(evidence),
            synthesis="",
            key_claims=[],
            gaps=[],
            experiments=[],
            citations=[],
        )
        synthesis, gaps, experiments, citations = _synthesize(evidence)
        report.synthesis = synthesis
        report.key_claims = _derive_key_claims(synthesis)
        report.gaps = gaps
        report.experiments = experiments
        report.citations = citations

    errors = validate_report_citations(report)
    report_text = report.synthesis + "\n" + "\n".join(report.gaps + [e.proposal for e in report.experiments])
    errors.extend(validate_no_new_numbers(report_text, evidence))
    errors.extend(validate_claim_support(report, evidence))
    if errors:
        raise ValueError("Validation failed: " + "; ".join(errors))
    return report


def render_report_markdown(report: ResearchReport, metrics: dict[str, float | int] | None = None) -> str:
    out = [f"# Research Report\n", f"## Question\n{report.question}\n"]
    out.append("## Paper Shortlist")
    for item in report.shortlist:
        out.append(f"- `{item.paper_id}`: {item.reason}")

    out.append("\n## Synthesis")
    out.append(report.synthesis)

    out.append("\n## Key Claims")
    if report.key_claims:
        for c in report.key_claims:
            out.append(f"- {c}")
    else:
        out.append("- None")

    out.append("\n## Gaps")
    for g in report.gaps:
        out.append(f"- {g}")

    out.append("\n## Experiment Proposals")
    for e in report.experiments:
        out.append(f"- {e.proposal}")

    out.append("\n## Citations")
    for c in sorted(set(report.citations)):
        out.append(f"- {c}")

    if report.retrieval_diagnostics:
        out.append("\n## Retrieval Diagnostics")
        for k, v in report.retrieval_diagnostics.items():
            out.append(f"- {k}: {v}")

    if metrics is not None:
        out.append("\n## Coverage Metrics")
        for k, v in metrics.items():
            out.append(f"- {k}: {v}")

    return "\n".join(out) + "\n"


def _truncate_tokens(text: str, max_tokens: int | None) -> str:
    if not max_tokens:
        return text
    toks = text.split()
    if len(toks) <= int(max_tokens):
        return text
    return " ".join(toks[: int(max_tokens)])


def _cache_key(
    index_path: str,
    question: str,
    top_k: int,
    min_items: int,
    min_score: float,
    retrieval_mode: str,
    alpha: float,
    max_per_paper: int,
    vector_index_path: str | None,
    vector_metadata_path: str | None,
    embedding_model: str,
    quality_prior_weight: float,
    sources_config_sig: str,
    metadata_prior_weight_val: float,
    max_summary_tokens: int,
    semantic_mode: str,
    semantic_model: str,
    semantic_min_support: float,
    semantic_max_contradiction: float,
    semantic_shadow_mode: bool,
    semantic_fail_on_low_support: bool,
    online_semantic_model: str,
    online_semantic_timeout_sec: float,
    online_semantic_max_checks: int,
    online_semantic_on_warn_only: bool,
    vector_service_endpoint: str | None,
    vector_nprobe: int,
    vector_ef_search: int,
    vector_topk_candidate_multiplier: float,
    live_enabled: bool,
    live_sources_override: str | None,
    live_max_items: int,
    live_timeout_sec: int,
    live_cache_ttl_sec: int | None,
    live_merge_mode: str,
    routing_mode: str,
    forced_intent: str | None,
    relevance_policy: str,
    diagnostics: bool,
    direct_answer_mode: str,
    direct_answer_max_complexity: int,
    use_kb: bool,
    kb_db: str | None,
    kb_top_k: int,
    kb_merge_weight: float,
) -> str:
    p = Path(index_path)
    vector_mtime_ns = 0
    vector_meta_mtime_ns = 0
    if vector_index_path:
        vp = Path(vector_index_path)
        if vp.exists():
            vector_mtime_ns = vp.stat().st_mtime_ns
    if vector_metadata_path:
        mp = Path(vector_metadata_path)
        if mp.exists():
            vector_meta_mtime_ns = mp.stat().st_mtime_ns
    payload = {
        "index": str(p.resolve()),
        "mtime_ns": p.stat().st_mtime_ns if p.exists() else 0,
        "question": question,
        "top_k": top_k,
        "min_items": min_items,
        "min_score": min_score,
        "domain": "computer_vision",
        "retrieval_mode": retrieval_mode,
        "alpha": alpha,
        "max_per_paper": max_per_paper,
        "vector_index_path": vector_index_path,
        "vector_index_mtime_ns": vector_mtime_ns,
        "vector_metadata_path": vector_metadata_path,
        "vector_metadata_mtime_ns": vector_meta_mtime_ns,
        "embedding_model": embedding_model,
        "quality_prior_weight": quality_prior_weight,
        "sources_config_sig": sources_config_sig,
        "metadata_prior_weight": metadata_prior_weight_val,
        "max_summary_tokens": max_summary_tokens,
        "semantic_mode": semantic_mode,
        "semantic_model": semantic_model,
        "semantic_min_support": semantic_min_support,
        "semantic_max_contradiction": semantic_max_contradiction,
        "semantic_shadow_mode": semantic_shadow_mode,
        "semantic_fail_on_low_support": semantic_fail_on_low_support,
        "online_semantic_model": online_semantic_model,
        "online_semantic_timeout_sec": online_semantic_timeout_sec,
        "online_semantic_max_checks": online_semantic_max_checks,
        "online_semantic_on_warn_only": online_semantic_on_warn_only,
        "vector_service_endpoint": vector_service_endpoint,
        "vector_nprobe": vector_nprobe,
        "vector_ef_search": vector_ef_search,
        "vector_topk_candidate_multiplier": vector_topk_candidate_multiplier,
        "live_enabled": live_enabled,
        "live_sources_override": live_sources_override,
        "live_max_items": live_max_items,
        "live_timeout_sec": live_timeout_sec,
        "live_cache_ttl_sec": live_cache_ttl_sec,
        "live_merge_mode": live_merge_mode,
        "routing_mode": routing_mode,
        "forced_intent": forced_intent,
        "relevance_policy": relevance_policy,
        "diagnostics": diagnostics,
        "direct_answer_mode": direct_answer_mode,
        "direct_answer_max_complexity": direct_answer_max_complexity,
        "use_kb": use_kb,
        "kb_db": kb_db,
        "kb_top_k": kb_top_k,
        "kb_merge_weight": kb_merge_weight,
    }
    return sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _query_vector_with_cfg(
    loaded_vector,
    *,
    question: str,
    top_k: int,
    embedding_model: str,
    vector_nprobe: int,
    vector_ef_search: int,
):
    try:
        return query_vector_index(
            loaded_vector,
            question=question,
            top_k=top_k,
            model_name_override=embedding_model,
            nprobe=vector_nprobe,
            ef_search=vector_ef_search,
        )
    except TypeError:
        # Backward compatibility for tests/mocks that still use older signature.
        return query_vector_index(
            loaded_vector,
            question=question,
            top_k=top_k,
            model_name_override=embedding_model,
        )


def _query_vector_via_service(
    endpoint: str,
    *,
    question: str,
    top_k: int,
    embedding_model: str,
    vector_nprobe: int,
    vector_ef_search: int,
) -> list[tuple[str, float]] | None:
    payload = json.dumps(
        {
            "question": question,
            "top_k": int(top_k),
            "embedding_model": embedding_model,
            "vector_nprobe": int(vector_nprobe),
            "vector_ef_search": int(vector_ef_search),
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        endpoint.rstrip("/") + "/query",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=2.0) as resp:
            body = resp.read().decode("utf-8")
    except (urllib.error.URLError, TimeoutError, ValueError):
        return None
    try:
        obj = json.loads(body)
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict):
        return None
    rows = obj.get("results", [])
    if not isinstance(rows, list):
        return None
    out: list[tuple[str, float]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        sid = row.get("snippet_id")
        score = row.get("score")
        if isinstance(sid, str) and isinstance(score, (int, float)):
            out.append((sid, float(score)))
    return out


def _write_outputs(out_path: str, report: ResearchReport, evidence: EvidencePack, metrics: dict[str, float | int]) -> str:
    md = render_report_markdown(report, metrics=metrics)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md, encoding="utf-8")
    out.with_suffix(".evidence.json").write_text(json.dumps(evidence.model_dump(), indent=2), encoding="utf-8")
    out.with_suffix(".json").write_text(json.dumps(report.model_dump(), indent=2), encoding="utf-8")
    out.with_suffix(".metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return str(out)


def _select_live_sources(
    *,
    question: str,
    configured: dict[str, object],
    override_csv: str | None,
) -> list[str]:
    if override_csv:
        return [x.strip() for x in override_csv.split(",") if x.strip() and x.strip() in configured]
    q = question.lower()
    selected: list[str] = []
    for name, row in configured.items():
        tags = [str(t).strip().lower() for t in (getattr(row, "domain_tags", []) or [])]
        if not tags:
            continue
        if any(tag and tag in q for tag in tags):
            selected.append(name)
    if selected:
        return selected
    return list(configured.keys())


def _fetch_live_snippets(
    *,
    question: str,
    sources_cfg: dict[str, object],
    selected_sources: list[str],
    snapshot_cfg: dict[str, object],
    live_max_items: int,
    live_timeout_sec: int,
    live_cache_ttl_sec: int | None,
    run_id: str,
) -> tuple[list[object], dict[str, int], list[str]]:
    snippets: list[object] = []
    artifacts: list[str] = []
    stats = {
        "live_fetch_attempted": 0,
        "live_fetch_succeeded": 0,
        "live_snapshot_count": 0,
        "live_snippet_count": 0,
        "live_cache_hit_count": 0,
        "live_fetch_error_count": 0,
    }
    run_live_dir = Path("runs/live") / run_id
    run_live_dir.mkdir(parents=True, exist_ok=True)
    for src_name in selected_sources:
        source = sources_cfg.get(src_name)
        if source is None:
            continue
        stats["live_fetch_attempted"] += 1
        params: dict[str, str] = {}
        ttl = int(live_cache_ttl_sec) if live_cache_ttl_sec is not None else int(getattr(source, "cache_ttl_sec", 300))
        try:
            cached = load_cached_snapshot(
                root_dir=str(snapshot_cfg.get("root_dir", "data/live_snapshots")),
                source=src_name,
                query=question,
                params=params,
                ttl_sec=ttl,
            )
            if cached is not None:
                snap = cached
                stats["live_cache_hit_count"] += 1
            else:
                items, raw = fetch_live_source(
                    source,
                    query=question,
                    params=params,
                    timeout_sec=int(live_timeout_sec),
                    max_items=int(live_max_items),
                    max_body_bytes=int(snapshot_cfg.get("max_body_bytes", 2_000_000)),
                )
                snap, _ = save_snapshot(
                    root_dir=str(snapshot_cfg.get("root_dir", "data/live_snapshots")),
                    source=src_name,
                    query=question,
                    params=params,
                    items=items,
                    persist_raw=bool(snapshot_cfg.get("persist_raw", True)),
                    raw_payload=raw,
                )
            source_snips = snapshot_to_snippets(snap, max_items=int(live_max_items), section="live")
            snippets.extend(source_snips)
            stats["live_snapshot_count"] += 1
            stats["live_fetch_succeeded"] += 1
            stats["live_snippet_count"] += len(source_snips)
            art = run_live_dir / f"live_fetch.{src_name}.json"
            art.write_text(
                json.dumps(
                    {
                        "source": src_name,
                        "question": question,
                        "snapshot_id": snap.snapshot_id,
                        "snippets": [s.model_dump() for s in source_snips],
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            artifacts.append(str(art))
        except Exception:
            stats["live_fetch_error_count"] += 1
            continue
    return snippets, stats, artifacts


def _merge_static_live_evidence(
    *,
    question: str,
    static_ev: EvidencePack,
    live_ev: EvidencePack,
    merge_mode: str,
    top_k: int,
    max_per_paper: int,
) -> EvidencePack:
    if not live_ev.items:
        return static_ev
    if merge_mode == "live_first":
        merged: list[EvidenceItem] = []
        seen: set[str] = set()
        for item in list(live_ev.items) + list(static_ev.items):
            if item.snippet_id in seen:
                continue
            merged.append(item)
            seen.add(item.snippet_id)
            if len(merged) >= top_k:
                break
        return EvidencePack(question=question, items=merged)

    static_scores = {it.snippet_id: float(it.score) for it in static_ev.items}
    live_scores = {it.snippet_id: float(it.score) for it in live_ev.items}
    s_norm = minmax_normalize_scores(static_scores)
    l_norm = minmax_normalize_scores(live_scores)
    combined: dict[str, tuple[EvidenceItem, float]] = {}
    for it in static_ev.items:
        combined[it.snippet_id] = (it, s_norm.get(it.snippet_id, 0.0))
    for it in live_ev.items:
        base = l_norm.get(it.snippet_id, 0.0)
        boosted = min(1.0, base * 1.05)
        combined[it.snippet_id] = (it, boosted)
    ranked = sorted(combined.values(), key=lambda x: x[1], reverse=True)
    out: list[EvidenceItem] = []
    per_paper: dict[str, int] = {}
    live_ids = {it.snippet_id for it in live_ev.items}
    best_live: EvidenceItem | None = None
    best_live_score = -1.0
    for it in live_ev.items:
        s = combined.get(it.snippet_id, (it, 0.0))[1]
        if s > best_live_score:
            best_live = it
            best_live_score = s
    for item, score in ranked:
        count = per_paper.get(item.paper_id, 0)
        if count >= max_per_paper:
            continue
        item.score = float(score)
        out.append(item)
        per_paper[item.paper_id] = count + 1
        if len(out) >= top_k:
            break
    if best_live is not None and out and not any(it.snippet_id in live_ids for it in out):
        best_live.score = float(best_live_score)
        out[-1] = best_live
    return EvidencePack(question=question, items=out)


def run_research(
    index_path: str,
    question: str,
    top_k: int,
    out_path: str,
    min_items: int = 2,
    min_score: float = 0.5,
    use_cache: bool = True,
    cache_dir: str = "runs/cache/research",
    retrieval_mode: str = "lexical",
    alpha: float = 0.6,
    max_per_paper: int = 2,
    vector_index_path: str | None = None,
    vector_metadata_path: str | None = None,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    quality_prior_weight: float = 0.15,
    sources_config_path: str | None = None,
    semantic_mode: str = "offline",
    semantic_model: str = DEFAULT_EMBEDDING_MODEL,
    semantic_min_support: float = 0.55,
    semantic_max_contradiction: float = 0.30,
    semantic_shadow_mode: bool = True,
    semantic_fail_on_low_support: bool = False,
    online_semantic_model: str = "gpt-4o-mini",
    online_semantic_timeout_sec: float = 12.0,
    online_semantic_max_checks: int = 12,
    online_semantic_on_warn_only: bool = True,
    online_semantic_base_url: str | None = None,
    online_semantic_api_key: str | None = None,
    vector_service_endpoint: str | None = None,
    vector_nprobe: int = 16,
    vector_ef_search: int = 64,
    vector_topk_candidate_multiplier: float = 1.5,
    live_enabled: bool = False,
    live_sources_override: str | None = None,
    live_max_items: int = 20,
    live_timeout_sec: int = 20,
    live_cache_ttl_sec: int | None = None,
    live_merge_mode: str = "union",
    routing_mode: str = "auto",
    intent: str | None = None,
    relevance_policy: str = "not_found",
    diagnostics: bool = True,
    direct_answer_mode: str = "hybrid",
    direct_answer_max_complexity: int = 2,
    use_kb: bool = False,
    kb_db: str = "data/kb/knowledge.db",
    kb_top_k: int = 5,
    kb_merge_weight: float = 0.15,
) -> str:
    if retrieval_mode not in {"lexical", "vector", "hybrid"}:
        raise ValueError("retrieval_mode must be one of: lexical, vector, hybrid")
    if live_merge_mode not in {"union", "live_first"}:
        raise ValueError("live_merge_mode must be one of: union, live_first")
    if routing_mode not in {"auto", "manual"}:
        raise ValueError("routing_mode must be one of: auto, manual")
    if relevance_policy not in {"not_found", "warn", "fail"}:
        raise ValueError("relevance_policy must be one of: not_found, warn, fail")
    if direct_answer_mode not in {"off", "hybrid"}:
        raise ValueError("direct_answer_mode must be one of: off, hybrid")

    t0 = time.perf_counter()
    vec_idx_path, vec_meta_path = derive_vector_paths(index_path, vector_index_path, vector_metadata_path)
    sources_cfg = load_sources_config(sources_config_path)
    if routing_mode == "auto":
        routing_mode = default_routing_mode(sources_cfg)
    md_prior_weight = metadata_prior_weight(sources_cfg, default=0.2)
    max_summary_tokens = max_tokens_per_summary(sources_cfg) or 0
    src_sig = ""
    if sources_config_path:
        p = Path(sources_config_path)
        src_sig = f"{str(p.resolve())}:{p.stat().st_mtime_ns}" if p.exists() else str(p)
    key = _cache_key(
        index_path,
        question,
        top_k,
        min_items,
        min_score,
        retrieval_mode=retrieval_mode,
        alpha=alpha,
        max_per_paper=max_per_paper,
        vector_index_path=vec_idx_path if retrieval_mode in {"vector", "hybrid"} else None,
        vector_metadata_path=vec_meta_path if retrieval_mode in {"vector", "hybrid"} else None,
        embedding_model=embedding_model,
        quality_prior_weight=quality_prior_weight,
        sources_config_sig=src_sig,
        metadata_prior_weight_val=md_prior_weight,
        max_summary_tokens=max_summary_tokens,
        semantic_mode=semantic_mode,
        semantic_model=semantic_model,
        semantic_min_support=semantic_min_support,
        semantic_max_contradiction=semantic_max_contradiction,
        semantic_shadow_mode=semantic_shadow_mode,
        semantic_fail_on_low_support=semantic_fail_on_low_support,
        online_semantic_model=online_semantic_model,
        online_semantic_timeout_sec=online_semantic_timeout_sec,
        online_semantic_max_checks=online_semantic_max_checks,
        online_semantic_on_warn_only=online_semantic_on_warn_only,
        vector_service_endpoint=vector_service_endpoint,
        vector_nprobe=vector_nprobe,
        vector_ef_search=vector_ef_search,
        vector_topk_candidate_multiplier=vector_topk_candidate_multiplier,
        live_enabled=live_enabled,
        live_sources_override=live_sources_override,
        live_max_items=live_max_items,
        live_timeout_sec=live_timeout_sec,
        live_cache_ttl_sec=live_cache_ttl_sec,
        live_merge_mode=live_merge_mode,
        routing_mode=routing_mode,
        forced_intent=intent,
        relevance_policy=relevance_policy,
        diagnostics=diagnostics,
        direct_answer_mode=direct_answer_mode,
        direct_answer_max_complexity=direct_answer_max_complexity,
        use_kb=use_kb,
        kb_db=kb_db,
        kb_top_k=kb_top_k,
        kb_merge_weight=kb_merge_weight,
    )

    direct = try_direct_answer(question, max_complexity=int(direct_answer_max_complexity)) if direct_answer_mode == "hybrid" else None
    if direct and direct.used:
        answer = str(direct.value)
        report = ResearchReport(
            question=question,
            shortlist=[],
            synthesis=f"Direct answer: {answer}",
            gaps=[],
            experiments=[],
            citations=[],
            retrieval_diagnostics={
                "direct_answer_used": True,
                "direct_answer_type": direct.answer_type,
                "direct_answer_value": answer,
            } if diagnostics else None,
        )
        evidence = EvidencePack(question=question, items=[])
        metrics: dict[str, Any] = {
            "cache_hit": False,
            "cache_key": key,
            "retrieval_mode": retrieval_mode,
            "live_enabled": bool(live_enabled),
            "intent_detected": "direct_answer",
            "intent_confidence": 1.0,
            "routing_mode": routing_mode,
            "routed_sources": [],
            "missing_intent_source_pack": False,
            "relevance_policy": relevance_policy,
            "relevance_reject_reason": "none",
            "diagnostics_emitted": bool(diagnostics),
            "direct_answer_used": True,
            "direct_answer_type": direct.answer_type,
            "direct_answer_value": answer,
            "kb_query_used": bool(use_kb),
            "kb_db": kb_db if use_kb else None,
            "kb_top_k": int(kb_top_k),
            "kb_merge_weight": float(kb_merge_weight),
            "kb_candidates_injected": 0,
            "question_relevance_gate_passed": True,
            "elapsed_ms_total": round((time.perf_counter() - t0) * 1000.0, 3),
        }
        return _write_outputs(out_path, report, evidence, metrics)
    cache_path = Path(cache_dir) / f"{key}.json"
    if use_cache and (not live_enabled) and cache_path.exists():
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        report = ResearchReport(**payload["report"])
        evidence = EvidencePack(**payload["evidence"])
        metrics = dict(payload["metrics"])
        cache_hit = True
        metrics["cache_hit"] = True
        metrics["cache_key"] = key
        metrics["elapsed_ms_index_load"] = 0.0
        metrics["elapsed_ms_query"] = 0.0
        metrics["elapsed_ms_report"] = 0.0
        metrics["elapsed_ms_total"] = round((time.perf_counter() - t0) * 1000.0, 3)
        return _write_outputs(out_path, report, evidence, metrics)

    t_index = time.perf_counter()
    index = load_index(index_path)
    t_query = time.perf_counter()
    plan = build_query_plan(question, domain="computer_vision")
    lexical_candidates = 0
    vector_candidates = 0
    vector_index_loaded = False
    vector_mismatch = False
    vector_quality_penalty_applied_count = 0
    live_sources_used: list[str] = []
    live_artifacts: list[str] = []
    live_snippets_count = 0
    kb_candidates_injected = 0
    kb_query_used = False
    kb_topic = intent if intent else None
    routed_sources: list[str] = []
    missing_intent_source_pack = False
    intent_result = IntentResult(intent="none", confidence=0.0, matched_terms=[])
    reject_reason = "none"
    diagnostics_payload: dict[str, Any] | None = None
    live_fetch_stats = {
        "live_fetch_attempted": 0,
        "live_fetch_succeeded": 0,
        "live_snapshot_count": 0,
        "live_snippet_count": 0,
        "live_cache_hit_count": 0,
        "live_fetch_error_count": 0,
    }
    pool_k = max(int(round(top_k * max(1.0, float(vector_topk_candidate_multiplier)))), max(top_k, 50))

    if retrieval_mode == "lexical":
        evidence = index.query(
            question,
            top_k=top_k,
            query_terms=plan.query_terms,
            term_boosts=plan.term_boosts,
            section_weights=plan.section_weights,
            max_per_paper=max_per_paper,
            metadata_prior_weight=md_prior_weight,
            quality_prior_weight=quality_prior_weight,
        )
        lexical_candidates = len(evidence.items)
    elif retrieval_mode == "vector":
        loaded_vector = load_vector_index(vec_idx_path, vec_meta_path)
        vector_index_loaded = True
        snippet_ids = [s.snippet_id for s in index.snippets]
        if len(loaded_vector.snippet_ids) != len(snippet_ids) or set(loaded_vector.snippet_ids) != set(snippet_ids):
            raise ValueError(
                "Vector metadata mismatch with lexical index snippets. Rebuild with build-index --with-vector."
            )
        vector_ranked = _query_vector_with_cfg(
            loaded_vector,
            question=question,
            top_k=pool_k,
            embedding_model=embedding_model,
            vector_nprobe=vector_nprobe,
            vector_ef_search=vector_ef_search,
        )
        if vector_service_endpoint:
            from_service = _query_vector_via_service(
                vector_service_endpoint,
                question=question,
                top_k=pool_k,
                embedding_model=embedding_model,
                vector_nprobe=vector_nprobe,
                vector_ef_search=vector_ef_search,
            )
            if from_service is not None:
                vector_ranked = from_service
        vector_candidates = len(vector_ranked)
        snippet_by_id = {s.snippet_id: s for s in index.snippets}
        raw_scores = {sid: score for sid, score in vector_ranked if sid in snippet_by_id}
        adj_scores = _apply_quality_to_vector_scores(raw_scores, snippet_by_id, quality_prior_weight=quality_prior_weight)
        vector_quality_penalty_applied_count = len(adj_scores)
        ranked = sorted(((snippet_by_id[sid], score) for sid, score in adj_scores.items()), key=lambda x: x[1], reverse=True)
        evidence = evidence_from_ranked(question, ranked, top_k=top_k, max_per_paper=max_per_paper)
    else:
        lexical_ranked = index.query_scored(
            question=question,
            top_k=pool_k,
            query_terms=plan.query_terms,
            term_boosts=plan.term_boosts,
            section_weights=plan.section_weights,
            max_per_paper=None,
            metadata_prior_weight=md_prior_weight,
            quality_prior_weight=quality_prior_weight,
        )
        lexical_scores = {sn.snippet_id: score for sn, score in lexical_ranked}
        lexical_candidates = len(lexical_scores)

        try:
            loaded_vector = load_vector_index(vec_idx_path, vec_meta_path)
            vector_index_loaded = True
            snippet_ids = [s.snippet_id for s in index.snippets]
            if len(loaded_vector.snippet_ids) != len(snippet_ids) or set(loaded_vector.snippet_ids) != set(snippet_ids):
                raise ValueError("Vector metadata mismatch with lexical index snippets")
            vector_ranked = _query_vector_with_cfg(
                loaded_vector,
                question=question,
                top_k=pool_k,
                embedding_model=embedding_model,
                vector_nprobe=vector_nprobe,
                vector_ef_search=vector_ef_search,
            )
            if vector_service_endpoint:
                from_service = _query_vector_via_service(
                    vector_service_endpoint,
                    question=question,
                    top_k=pool_k,
                    embedding_model=embedding_model,
                    vector_nprobe=vector_nprobe,
                    vector_ef_search=vector_ef_search,
                )
                if from_service is not None:
                    vector_ranked = from_service
            snippet_by_id = {s.snippet_id: s for s in index.snippets}
            raw_vector_scores = {sid: score for sid, score in vector_ranked if sid in snippet_by_id}
            vector_scores = _apply_quality_to_vector_scores(
                raw_vector_scores,
                snippet_by_id,
                quality_prior_weight=quality_prior_weight,
            )
            vector_quality_penalty_applied_count = len(vector_scores)
            vector_candidates = len(vector_scores)
            fused = fuse_hybrid_scores(lexical_scores, vector_scores, alpha=alpha)
            ranked = sorted(
                ((snippet_by_id[sid], score) for sid, score in fused.items() if sid in snippet_by_id),
                key=lambda x: x[1],
                reverse=True,
            )
            evidence = evidence_from_ranked(question, ranked, top_k=top_k, max_per_paper=max_per_paper)
        except ValueError:
            vector_mismatch = True
            fallback_ranked = sorted(lexical_ranked, key=lambda x: x[1], reverse=True)
            evidence = evidence_from_ranked(question, fallback_ranked, top_k=top_k, max_per_paper=max_per_paper)

    if live_enabled:
        live_cfg = live_sources_config(sources_cfg)
        enabled_live = {name: row for name, row in live_cfg.items() if bool(getattr(row, "enabled", False))}
        if live_sources_override:
            selected = _select_live_sources(question=question, configured=enabled_live, override_csv=live_sources_override)
            intent_result = IntentResult(intent="manual_override", confidence=1.0, matched_terms=[])
        elif routing_mode == "manual":
            selected = []
            intent_result = IntentResult(intent="manual", confidence=1.0, matched_terms=[])
        elif not has_live_routing(sources_cfg):
            selected = _select_live_sources(question=question, configured=enabled_live, override_csv=None)
            intent_result = IntentResult(intent="legacy_tags", confidence=0.5 if selected else 0.0, matched_terms=[])
        else:
            intent_result = detect_intent(question, sources_cfg, forced_intent=intent)
            selected = sources_for_intent(intent_result.intent, sources_cfg, enabled_live)
            if not selected:
                missing_intent_source_pack = True
                reject_reason = "no_sources_for_intent"
                diagnostics_payload = build_not_found_diagnostics(
                    question=question,
                    intent_result=intent_result,
                    attempted_sources=[],
                    reject_reason=reject_reason,
                    cfg=sources_cfg,
                )
                evidence = EvidencePack(question=question, items=[])
        routed_sources = list(selected)
        live_sources_used = selected
        if selected:
            snap_cfg = live_snapshot_config(sources_cfg)
            run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            live_snippets, live_fetch_stats, live_artifacts = _fetch_live_snippets(
                question=question,
                sources_cfg=enabled_live,
                selected_sources=selected,
                snapshot_cfg=snap_cfg,
                live_max_items=int(live_max_items),
                live_timeout_sec=int(live_timeout_sec),
                live_cache_ttl_sec=live_cache_ttl_sec,
                run_id=run_id,
            )
            live_snippets_count = len(live_snippets)
            if not live_snippets and reject_reason == "none":
                reject_reason = "empty_live_results"
                diagnostics_payload = build_not_found_diagnostics(
                    question=question,
                    intent_result=intent_result,
                    attempted_sources=selected,
                    reject_reason=reject_reason,
                    cfg=sources_cfg,
                )
            if live_snippets:
                live_index = SimpleBM25Index.build(live_snippets)
                live_ranked = live_index.query_scored(
                    question=question,
                    top_k=max(int(live_max_items), top_k),
                    query_terms=plan.query_terms,
                    term_boosts=plan.term_boosts,
                    section_weights=plan.section_weights,
                    max_per_paper=max_per_paper,
                    metadata_prior_weight=0.0,
                    quality_prior_weight=0.0,
                )
                if not live_ranked:
                    live_ranked = [(sn, 0.01) for sn in live_snippets[: max(int(live_max_items), top_k)]]
                trust_by_source = {name: float(getattr(cfg_row, "source_trust", 1.0)) for name, cfg_row in enabled_live.items()}
                adj_live_ranked = []
                for sn, score in live_ranked:
                    src = str(sn.paper_venue or "").strip()
                    trust = trust_by_source.get(src, 1.0)
                    adj_live_ranked.append((sn, float(score) * min(1.25, max(0.5, trust))))
                live_evidence = evidence_from_ranked(
                    question,
                    sorted(adj_live_ranked, key=lambda x: x[1], reverse=True),
                    top_k=max(int(live_max_items), top_k),
                    max_per_paper=max_per_paper,
                )
                evidence = _merge_static_live_evidence(
                    question=question,
                    static_ev=evidence,
                    live_ev=live_evidence,
                    merge_mode=live_merge_mode,
                    top_k=top_k,
                    max_per_paper=max_per_paper,
                )

    if use_kb:
        kb_query_used = True
        try:
            kb_candidates = query_kb_as_evidence(
                kb_db=kb_db,
                question=question,
                topic=kb_topic,
                top_k=int(kb_top_k),
                merge_weight=float(kb_merge_weight),
            )
        except Exception:
            kb_candidates = []
        kb_candidates_injected = len(kb_candidates)
        if kb_candidates:
            by_id: dict[str, EvidenceItem] = {it.snippet_id: it for it in evidence.items}
            for it in kb_candidates:
                cur = by_id.get(it.snippet_id)
                if cur is None or float(it.score) > float(cur.score):
                    by_id[it.snippet_id] = it
            merged_ranked = sorted(by_id.values(), key=lambda x: float(x.score), reverse=True)
            per_paper: dict[str, int] = {}
            trimmed: list[EvidenceItem] = []
            for it in merged_ranked:
                cnt = per_paper.get(it.paper_id, 0)
                if cnt >= max_per_paper:
                    continue
                trimmed.append(it)
                per_paper[it.paper_id] = cnt + 1
                if len(trimmed) >= top_k:
                    break
            evidence = EvidencePack(question=question, items=trimmed)

    relevance = _question_evidence_relevance(question, evidence)
    relevance_threshold = 0.18
    relevance_gate_passed = bool(
        (float(relevance.get("question_relevance_max", 0.0)) >= relevance_threshold)
        or (int(relevance.get("question_shared_terms_max", 0)) >= 1 and float(relevance.get("question_relevance_avg", 0.0)) >= 0.08)
    )
    if evidence.items and not relevance_gate_passed and reject_reason == "none":
        reject_reason = "low_relevance"
        diagnostics_payload = build_not_found_diagnostics(
            question=question,
            intent_result=intent_result,
            attempted_sources=routed_sources or live_sources_used,
            reject_reason=reject_reason,
            cfg=sources_cfg,
        )
    if evidence.items and not relevance_gate_passed and relevance_policy == "not_found":
        evidence = EvidencePack(question=question, items=[])
    if relevance_policy == "fail" and reject_reason != "none":
        raise ValueError(
            f"relevance_policy_fail: {reject_reason}; intent={intent_result.intent}; "
            f"attempted_sources={','.join(routed_sources or live_sources_used)}"
        )

    t_report = time.perf_counter()
    report = build_research_report(question, evidence, min_items=min_items, min_score=min_score)
    if diagnostics and diagnostics_payload:
        report.retrieval_diagnostics = diagnostics_payload
    elif diagnostics:
        report.retrieval_diagnostics = {
            "intent_detected": intent_result.intent,
            "intent_confidence": float(intent_result.confidence),
            "attempted_sources": routed_sources or live_sources_used,
            "reject_reason": reject_reason,
        }
    if max_summary_tokens > 0:
        report.synthesis = _truncate_tokens(report.synthesis, max_summary_tokens)
        report.gaps = [_truncate_tokens(g, max_summary_tokens) for g in report.gaps]
        report.experiments = [
            ExperimentProposal(proposal=_truncate_tokens(e.proposal, max_summary_tokens), citations=e.citations)
            for e in report.experiments
        ]
    metrics = report_coverage_metrics(report, evidence)
    sem_errors, sem_metrics, sem_warnings = validate_semantic_claim_support(
        report,
        evidence,
        semantic_mode=semantic_mode,
        model_name=semantic_model,
        min_support=semantic_min_support,
        max_contradiction=semantic_max_contradiction,
        shadow_mode=semantic_shadow_mode,
        fail_on_low_support=semantic_fail_on_low_support,
        online_model=online_semantic_model,
        online_timeout_sec=online_semantic_timeout_sec,
        online_max_checks=online_semantic_max_checks,
        online_on_warn_only=online_semantic_on_warn_only,
        online_base_url=online_semantic_base_url,
        online_api_key=online_semantic_api_key,
    )
    # Research path is non-blocking for semantic validation in this phase.
    if sem_errors:
        sem_metrics["semantic_status"] = "warn"
    if sem_warnings:
        sem_metrics["semantic_warnings_count"] = len(sem_warnings)
    metrics.update(sem_metrics)
    metrics["cache_hit"] = False
    metrics["cache_key"] = key
    metrics["retrieval_mode"] = retrieval_mode
    metrics["alpha"] = alpha
    metrics["max_per_paper"] = max_per_paper
    metrics["vector_index_loaded"] = vector_index_loaded
    metrics["vector_mismatch"] = vector_mismatch
    metrics["vector_candidates"] = vector_candidates
    metrics["vector_quality_penalty_applied_count"] = vector_quality_penalty_applied_count
    metrics["vector_nprobe"] = int(vector_nprobe)
    metrics["vector_ef_search"] = int(vector_ef_search)
    metrics["vector_topk_candidate_multiplier"] = float(vector_topk_candidate_multiplier)
    metrics["vector_service_endpoint"] = vector_service_endpoint
    metrics["lexical_candidates"] = lexical_candidates
    metrics["live_enabled"] = bool(live_enabled)
    metrics["live_sources_used"] = live_sources_used
    metrics["intent_detected"] = intent_result.intent
    metrics["intent_confidence"] = float(intent_result.confidence)
    metrics["routing_mode"] = routing_mode
    metrics["routed_sources"] = routed_sources
    metrics["missing_intent_source_pack"] = bool(missing_intent_source_pack)
    metrics["relevance_policy"] = relevance_policy
    metrics["relevance_reject_reason"] = reject_reason
    metrics["diagnostics_emitted"] = bool(diagnostics and diagnostics_payload is not None)
    metrics["direct_answer_used"] = False
    metrics["direct_answer_type"] = "none"
    metrics["direct_answer_value"] = None
    metrics["kb_query_used"] = bool(kb_query_used)
    metrics["kb_db"] = kb_db if use_kb else None
    metrics["kb_top_k"] = int(kb_top_k)
    metrics["kb_merge_weight"] = float(kb_merge_weight)
    metrics["kb_candidates_injected"] = int(kb_candidates_injected)
    metrics["live_merge_mode"] = live_merge_mode
    metrics["live_snippet_count"] = int(live_snippets_count)
    metrics["live_artifacts"] = live_artifacts
    metrics.update({k: int(v) for k, v in live_fetch_stats.items()})
    metrics.update(relevance)
    metrics["question_relevance_threshold"] = relevance_threshold
    metrics["question_relevance_gate_passed"] = bool(relevance_gate_passed)
    metrics["quality_prior_weight"] = quality_prior_weight
    metrics["metadata_prior_weight"] = round(md_prior_weight, 4)
    metrics["sources_config_applied"] = int(bool(sources_cfg))
    metrics["max_tokens_per_summary"] = max_summary_tokens
    selected_quality_scores = [float(i.extraction_quality_score) for i in evidence.items if i.extraction_quality_score is not None]
    metrics["quality_penalty_applied_count"] = len(selected_quality_scores)
    metrics["avg_quality_score_selected"] = round(sum(selected_quality_scores) / len(selected_quality_scores), 4) if selected_quality_scores else None
    metrics["elapsed_ms_index_load"] = round((t_query - t_index) * 1000.0, 3)
    metrics["elapsed_ms_query"] = round((t_report - t_query) * 1000.0, 3)
    metrics["elapsed_ms_report"] = round((time.perf_counter() - t_report) * 1000.0, 3)
    metrics["elapsed_ms_total"] = round((time.perf_counter() - t0) * 1000.0, 3)
    metrics.update({f"index_cache_{k}": v for k, v in index_cache_info().items()})

    out = _write_outputs(out_path, report, evidence, metrics)

    if use_cache and (not live_enabled):
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(
            json.dumps({"report": report.model_dump(), "evidence": evidence.model_dump(), "metrics": metrics}, indent=2),
            encoding="utf-8",
        )
    return out
