from __future__ import annotations

from pathlib import Path
import json
import statistics
import time

from src.apps.evidence_extract import load_snippets
from src.apps.research_copilot import build_research_report
from src.apps.query_builder import build_query_plan
from src.apps.retrieval import SimpleBM25Index, clear_index_cache, index_cache_info, load_index
from src.apps.retrieval import derive_vector_paths
from src.apps.vector_index import LoadedVectorIndex, build_vector_index, load_vector_index, query_vector_index
from src.core.ids import snippet_id
from src.core.schemas import SnippetRecord


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    vals = sorted(values)
    idx = int((len(vals) - 1) * pct)
    return vals[idx]


def _vector_eval_gate(
    recall_at_k: float | None,
    ann_p95_ms: float | None,
    exact_p95_ms: float | None,
    min_recall_ratio: float = 0.98,
    min_p95_improvement: float = 0.30,
) -> dict[str, object]:
    recall_ok = None if recall_at_k is None else bool(recall_at_k >= min_recall_ratio)
    speedup = None
    speed_ok = None
    if ann_p95_ms is not None and exact_p95_ms and exact_p95_ms > 0:
        speedup = max(0.0, (exact_p95_ms - ann_p95_ms) / exact_p95_ms)
        speed_ok = bool(speedup >= min_p95_improvement)
    passed = bool(recall_ok is True and speed_ok is True) if recall_ok is not None and speed_ok is not None else False
    return {
        "gate_enabled": recall_at_k is not None and ann_p95_ms is not None and exact_p95_ms is not None,
        "min_recall_ratio": min_recall_ratio,
        "min_p95_improvement": min_p95_improvement,
        "recall_ok": recall_ok,
        "speed_ok": speed_ok,
        "p95_speedup_ratio": None if speedup is None else round(speedup, 4),
        "pass": passed,
    }


def benchmark_research(
    index_path: str,
    queries: list[str],
    runs_per_query: int = 3,
    top_k: int = 8,
    min_items: int = 2,
    min_score: float = 0.5,
    retrieval_mode: str = "lexical",
    alpha: float = 0.6,
    max_per_paper: int = 2,
    vector_index_path: str | None = None,
    vector_metadata_path: str | None = None,
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    quality_prior_weight: float = 0.15,
    sources_config_path: str | None = None,
    semantic_mode: str = "offline",
    semantic_model: str = "sentence-transformers/all-MiniLM-L6-v2",
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
    out_path: str = "runs/research_reports/benchmark.json",
) -> str:
    if not queries:
        raise ValueError("queries cannot be empty")

    clear_index_cache()
    t0 = time.perf_counter()
    idx_loaded = load_index(index_path)
    cold_ms = (time.perf_counter() - t0) * 1000.0

    t1 = time.perf_counter()
    _ = load_index(index_path)
    warm_ms = (time.perf_counter() - t1) * 1000.0
    corpus_snippets = len(idx_loaded.snippets)
    if corpus_snippets < 50_000:
        corpus_bucket = "<50k"
    elif corpus_snippets < 200_000:
        corpus_bucket = "50k-200k"
    elif corpus_snippets < 1_000_000:
        corpus_bucket = "200k-1m"
    else:
        corpus_bucket = ">=1m"

    latencies_no_cache: list[float] = []
    latencies_with_cache: list[float] = []
    vector_eval_recall: list[float] = []
    vector_eval_ann_lat_ms: list[float] = []
    vector_eval_exact_lat_ms: list[float] = []
    vector_eval_enabled = retrieval_mode in {"vector", "hybrid"}
    loaded_current = None
    loaded_exact = None
    if vector_eval_enabled:
        vec_idx_path, vec_meta_path = derive_vector_paths(index_path, vector_index_path, vector_metadata_path)
        try:
            lexical_idx = load_index(index_path)
            loaded_current = load_vector_index(vec_idx_path, vec_meta_path)
            exact_bundle = build_vector_index(
                lexical_idx.snippets,
                model_name=embedding_model,
                index_type="flat",
                batch_size=64,
            )
            loaded_exact = LoadedVectorIndex(
                index=exact_bundle.index,
                snippet_ids=list(exact_bundle.snippet_ids),
                embedding_model=exact_bundle.embedding_model,
                dimension=exact_bundle.dimension,
                metric="cosine_via_ip",
                index_mtime_ns=0,
                index_type="flat",
                ann_params={},
                shard_count=1,
                shard_paths=[],
            )
        except Exception:
            vector_eval_enabled = False

    bench_dir = Path(out_path).parent / "benchmark_runs"
    bench_dir.mkdir(parents=True, exist_ok=True)

    from src.apps.research_copilot import run_research

    for qi, q in enumerate(queries):
        if vector_eval_enabled and loaded_current is not None and loaded_exact is not None:
            t_ann = time.perf_counter()
            ann_ranked = query_vector_index(
                loaded_current,
                question=q,
                top_k=top_k,
                model_name_override=embedding_model,
                nprobe=vector_nprobe,
                ef_search=vector_ef_search,
            )
            vector_eval_ann_lat_ms.append((time.perf_counter() - t_ann) * 1000.0)
            t_exact = time.perf_counter()
            exact_ranked = query_vector_index(
                loaded_exact,
                question=q,
                top_k=top_k,
                model_name_override=embedding_model,
            )
            vector_eval_exact_lat_ms.append((time.perf_counter() - t_exact) * 1000.0)
            ann_ids = [sid for sid, _ in ann_ranked]
            exact_ids = [sid for sid, _ in exact_ranked]
            if exact_ids:
                recall = len(set(ann_ids) & set(exact_ids)) / float(len(exact_ids))
                vector_eval_recall.append(recall)

        for ri in range(runs_per_query):
            out_a = bench_dir / f"q{qi}_r{ri}_nocache.md"
            t = time.perf_counter()
            run_research(
                index_path=index_path,
                question=q,
                top_k=top_k,
                out_path=str(out_a),
                min_items=min_items,
                min_score=min_score,
                use_cache=False,
                retrieval_mode=retrieval_mode,
                alpha=alpha,
                max_per_paper=max_per_paper,
                vector_index_path=vector_index_path,
                vector_metadata_path=vector_metadata_path,
                embedding_model=embedding_model,
                quality_prior_weight=quality_prior_weight,
                sources_config_path=sources_config_path,
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
                online_semantic_base_url=online_semantic_base_url,
                online_semantic_api_key=online_semantic_api_key,
                vector_service_endpoint=vector_service_endpoint,
                vector_nprobe=vector_nprobe,
                vector_ef_search=vector_ef_search,
                vector_topk_candidate_multiplier=vector_topk_candidate_multiplier,
            )
            latencies_no_cache.append((time.perf_counter() - t) * 1000.0)

            out_b = bench_dir / f"q{qi}_r{ri}_cache.md"
            t = time.perf_counter()
            run_research(
                index_path=index_path,
                question=q,
                top_k=top_k,
                out_path=str(out_b),
                min_items=min_items,
                min_score=min_score,
                use_cache=True,
                retrieval_mode=retrieval_mode,
                alpha=alpha,
                max_per_paper=max_per_paper,
                vector_index_path=vector_index_path,
                vector_metadata_path=vector_metadata_path,
                embedding_model=embedding_model,
                quality_prior_weight=quality_prior_weight,
                sources_config_path=sources_config_path,
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
                online_semantic_base_url=online_semantic_base_url,
                online_semantic_api_key=online_semantic_api_key,
                vector_service_endpoint=vector_service_endpoint,
                vector_nprobe=vector_nprobe,
                vector_ef_search=vector_ef_search,
                vector_topk_candidate_multiplier=vector_topk_candidate_multiplier,
            )
            latencies_with_cache.append((time.perf_counter() - t) * 1000.0)

    payload = {
        "index_path": str(Path(index_path).resolve()),
        "corpus_snippets": corpus_snippets,
        "corpus_size_bucket": corpus_bucket,
        "num_queries": len(queries),
        "runs_per_query": runs_per_query,
        "samples": len(latencies_no_cache),
        "retrieval_mode": retrieval_mode,
        "alpha": alpha,
        "max_per_paper": max_per_paper,
        "quality_prior_weight": quality_prior_weight,
        "vector_service_endpoint": vector_service_endpoint,
        "vector_nprobe": vector_nprobe,
        "vector_ef_search": vector_ef_search,
        "vector_topk_candidate_multiplier": vector_topk_candidate_multiplier,
        "sources_config_path": sources_config_path,
        "semantic_mode": semantic_mode,
        "cold_index_load_ms": round(cold_ms, 3),
        "warm_index_load_ms": round(warm_ms, 3),
        "latency_no_cache": {
            "mean_ms": round(statistics.mean(latencies_no_cache), 3),
            "p50_ms": round(_percentile(latencies_no_cache, 0.5), 3),
            "p95_ms": round(_percentile(latencies_no_cache, 0.95), 3),
        },
        "latency_with_cache": {
            "mean_ms": round(statistics.mean(latencies_with_cache), 3),
            "p50_ms": round(_percentile(latencies_with_cache, 0.5), 3),
            "p95_ms": round(_percentile(latencies_with_cache, 0.95), 3),
        },
        "index_cache": index_cache_info(),
        "vector_eval": {
            "enabled": vector_eval_enabled,
            "recall_at_k": round(statistics.mean(vector_eval_recall), 4) if vector_eval_recall else None,
            "ann_latency": {
                "p50_ms": round(_percentile(vector_eval_ann_lat_ms, 0.5), 3) if vector_eval_ann_lat_ms else None,
                "p95_ms": round(_percentile(vector_eval_ann_lat_ms, 0.95), 3) if vector_eval_ann_lat_ms else None,
            },
            "exact_latency": {
                "p50_ms": round(_percentile(vector_eval_exact_lat_ms, 0.5), 3) if vector_eval_exact_lat_ms else None,
                "p95_ms": round(_percentile(vector_eval_exact_lat_ms, 0.95), 3) if vector_eval_exact_lat_ms else None,
            },
        },
    }
    if latencies_no_cache:
        payload["qps_no_cache"] = round(1000.0 / statistics.mean(latencies_no_cache), 4)
    if latencies_with_cache:
        payload["qps_with_cache"] = round(1000.0 / statistics.mean(latencies_with_cache), 4)
    payload["vector_eval"]["gate"] = _vector_eval_gate(
        recall_at_k=payload["vector_eval"]["recall_at_k"],
        ann_p95_ms=payload["vector_eval"]["ann_latency"]["p95_ms"],
        exact_p95_ms=payload["vector_eval"]["exact_latency"]["p95_ms"],
    )

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return str(out)


def _expand_snippets(snippets: list[SnippetRecord], repeat_factor: int) -> list[SnippetRecord]:
    if repeat_factor < 1:
        raise ValueError("repeat_factor must be >= 1")
    expanded: list[SnippetRecord] = []
    for r in range(repeat_factor):
        for s in snippets:
            pid = f"{s.paper_id}__rep{r}"
            expanded.append(
                SnippetRecord(
                    snippet_id=snippet_id(pid, 1),
                    paper_id=pid,
                    section=s.section,
                    text=s.text,
                    page_hint=s.page_hint,
                    token_count=s.token_count,
                    paper_year=s.paper_year,
                    paper_venue=s.paper_venue,
                    citation_count=s.citation_count,
                )
            )
    return expanded


def benchmark_scale(
    corpus_dir: str,
    queries: list[str],
    repeat_factor: int = 20,
    runs_per_query: int = 2,
    top_k: int = 8,
    out_path: str = "runs/research_reports/benchmark_scale.json",
) -> str:
    if not queries:
        raise ValueError("queries cannot be empty")

    base = load_snippets(corpus_dir)
    expanded = _expand_snippets(base, repeat_factor)

    t_build0 = time.perf_counter()
    index = SimpleBM25Index.build(expanded)
    build_ms = (time.perf_counter() - t_build0) * 1000.0

    latencies: list[float] = []
    for q in queries:
        plan = build_query_plan(q, domain="computer_vision")
        for _ in range(runs_per_query):
            t = time.perf_counter()
            evidence = index.query(
                q,
                top_k=top_k,
                query_terms=plan.query_terms,
                term_boosts=plan.term_boosts,
                section_weights=plan.section_weights,
                max_per_paper=2,
            )
            _ = build_research_report(q, evidence, min_items=1, min_score=0.0)
            latencies.append((time.perf_counter() - t) * 1000.0)

    payload = {
        "base_snippets": len(base),
        "repeat_factor": repeat_factor,
        "expanded_snippets": len(expanded),
        "queries": len(queries),
        "runs_per_query": runs_per_query,
        "index_build_ms": round(build_ms, 3),
        "latency_ms": {
            "mean": round(statistics.mean(latencies), 3),
            "p50": round(_percentile(latencies, 0.5), 3),
            "p95": round(_percentile(latencies, 0.95), 3),
        },
    }

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return str(out)
