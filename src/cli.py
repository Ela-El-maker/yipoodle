from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys
from datetime import datetime, timezone
import time
from typing import Any

log = logging.getLogger(__name__)

from src.apps.doc_writer import write_doc
from src.apps.domain_scaffold import scaffold_domain_config
from src.apps.extraction_eval import scaffold_extraction_gold, write_extraction_eval_report
from src.apps.extraction_quality_report import write_extraction_quality_report
from src.apps.evidence_extract import extract_from_papers_dir, extract_from_papers_dir_with_db
from src.apps.index_builder import build_index, build_index_incremental
from src.apps.benchmark import benchmark_research, benchmark_scale
from src.apps.benchmark_regression import run_benchmark_regression_check
from src.apps.live_snapshot_store import load_cached_snapshot, save_snapshot, snapshot_to_snippets
from src.apps.live_sources import fetch_live_source, live_snapshot_config, live_sources_config
from src.apps.ask_mode import run_ask_mode
from src.apps.kb_diff import run_kb_diff
from src.apps.kb_contradiction_resolver import run_kb_contradiction_resolver
from src.apps.kb_ingest import backfill_kb, ingest_report_to_kb
from src.apps.kb_query import run_kb_query
from src.apps.layout_promotion import run_layout_promotion_gate
from src.apps.monitoring_engine import evaluate_topic_monitoring, monitor_digest_flush, monitor_status
from src.apps.monitoring_history import run_monitor_history_check
from src.apps.monitoring_soak import run_monitor_soak_sim
from src.apps.monitor_mode import run_monitor_mode, unregister_monitor
from src.apps.notes_mode import run_notes_mode
from src.apps.research_template import run_research_template
from src.apps.query_router_eval import run_query_router_eval
from src.apps.reliability_watchdog import run_reliability_watchdog
from src.apps.corpus_health import evaluate_corpus_health
from src.apps.corpus_migration import migrate_extraction_meta
from src.apps.snapshot import create_snapshot
from src.apps.paper_sync import sync_papers
from src.apps.pipeline_runner import run_full_pipeline
from src.apps.research_copilot import run_research
from src.apps.release_notes import generate_release_notes
from src.apps.structured_export import (
    export_report,
    export_report_to_file,
    export_report_multi,
    load_evidence_json,
    load_report_json,
)
from src.apps.session_store import (
    DEFAULT_SESSION_DB,
    create_session,
    close_session,
    reopen_session,
    delete_session,
    get_session_detail,
    list_sessions,
    record_query,
    render_session_markdown,
    session_summary,
)
from src.apps.source_reliability import (
    DEFAULT_RELIABILITY_DB,
    record_feedback,
    recompute_reliability,
    recompute_all,
    get_reliability,
    list_source_reliability,
    get_feedback_events,
    delete_source,
    reliability_report,
    render_reliability_markdown,
    EVENT_TYPES,
)
from src.apps.sources_config import load_sources_config, ocr_config
from src.apps.automation import dispatch_alerts, load_automation_config, run_automation
from src.apps.query_router import dispatch_query, load_router_config, route_query
from src.apps.vector_service import (
    start_vector_service_server,
    vector_service_build,
    vector_service_health,
    vector_service_query,
)
from src.apps.watch_ingest import run_watch_ingest
from src.core.schemas import EvidencePack
from src.core.validation import (
    report_coverage_metrics,
    validate_claim_support,
    validate_no_new_numbers,
    validate_report_citations,
    validate_semantic_claim_support,
)


def cmd_sync_papers(args: argparse.Namespace) -> None:
    stats = sync_papers(
        args.query,
        args.max_results,
        args.db_path,
        args.papers_dir,
        with_semantic_scholar=args.with_semantic_scholar,
        prefer_arxiv=args.prefer_arxiv,
        require_pdf=args.require_pdf,
        sources_config_path=args.sources_config,
    )
    print(json.dumps(stats, indent=2))
    if args.fail_on_source_quality_gate and not bool(stats.get("source_quality_healthy", True)):
        reasons = stats.get("source_quality_reasons", []) or []
        raise SystemExit("Source quality gate failed:\n- " + "\n- ".join(str(x) for x in reasons))


def cmd_build_index(args: argparse.Namespace) -> None:
    build_fn = build_index_incremental if getattr(args, "incremental", False) else build_index
    stats = build_fn(
        args.corpus,
        args.out,
        db_path=args.db_path,
        with_vector=args.with_vector,
        embedding_model=args.embedding_model,
        vector_index_path=args.vector_index_path,
        vector_metadata_path=args.vector_metadata_path,
        batch_size=args.batch_size,
        vector_index_type=args.vector_index_type,
        vector_nlist=args.vector_nlist,
        vector_m=args.vector_m,
        vector_ef_construction=args.vector_ef_construction,
        vector_shards=args.vector_shards,
        vector_train_sample_size=args.vector_train_sample_size,
        require_healthy_corpus=args.require_healthy_corpus,
        min_snippets=args.min_snippets,
        min_avg_chars_per_paper=args.min_avg_chars_per_paper,
        min_avg_chars_per_page=args.min_avg_chars_per_page,
        max_extract_error_rate=args.max_extract_error_rate,
        live_data_paths=args.live_data,
    )
    print(json.dumps(stats, indent=2))


def cmd_extract_corpus(args: argparse.Namespace) -> None:
    sources_cfg = load_sources_config(args.sources_config)
    cfg_ocr = ocr_config(sources_cfg)

    def _flag_present(*names: str) -> bool:
        return any(n in sys.argv for n in names)

    defaults = {
        "enabled": False,
        "timeout_sec": 30,
        "min_chars_trigger": 120,
        "max_pages": 20,
        "min_output_chars": 200,
        "min_gain_chars": 40,
        "min_confidence": 45.0,
        "lang": "eng",
        "profile": "document",
        "noise_suppression": True,
    }
    resolved_ocr_enabled = (
        bool(args.ocr_enabled)
        if _flag_present("--ocr-enabled", "--no-ocr-enabled")
        else bool(cfg_ocr.get("enabled", defaults["enabled"]))
    )
    resolved_ocr_timeout_sec = (
        int(args.ocr_timeout_sec)
        if args.ocr_timeout_sec is not None
        else int(cfg_ocr.get("timeout_sec", defaults["timeout_sec"]))
    )
    resolved_ocr_min_chars_trigger = (
        int(args.ocr_min_chars_trigger)
        if args.ocr_min_chars_trigger is not None
        else int(cfg_ocr.get("min_chars_trigger", defaults["min_chars_trigger"]))
    )
    resolved_ocr_max_pages = (
        int(args.ocr_max_pages)
        if args.ocr_max_pages is not None
        else int(cfg_ocr.get("max_pages", defaults["max_pages"]))
    )
    resolved_ocr_min_output_chars = (
        int(args.ocr_min_output_chars)
        if args.ocr_min_output_chars is not None
        else int(cfg_ocr.get("min_output_chars", defaults["min_output_chars"]))
    )
    resolved_ocr_min_gain_chars = (
        int(args.ocr_min_gain_chars)
        if args.ocr_min_gain_chars is not None
        else int(cfg_ocr.get("min_gain_chars", defaults["min_gain_chars"]))
    )
    resolved_ocr_min_confidence = (
        float(args.ocr_min_confidence)
        if args.ocr_min_confidence is not None
        else float(cfg_ocr.get("min_confidence", defaults["min_confidence"]))
    )
    resolved_ocr_lang = str(args.ocr_lang).strip() if args.ocr_lang is not None else str(cfg_ocr.get("lang", defaults["lang"]))
    resolved_ocr_profile = (
        str(args.ocr_profile).strip() if args.ocr_profile is not None else str(cfg_ocr.get("profile", defaults["profile"]))
    )
    resolved_ocr_noise_suppression = (
        bool(args.ocr_noise_suppression)
        if _flag_present("--ocr-noise-suppression", "--no-ocr-noise-suppression")
        else bool(cfg_ocr.get("noise_suppression", defaults["noise_suppression"]))
    )
    resolved_layout_engine = str(args.layout_engine or "shadow")
    if resolved_layout_engine == "auto":
        state_path = Path(args.layout_promotion_state)
        if state_path.exists():
            try:
                st = json.loads(state_path.read_text(encoding="utf-8"))
                if bool(st.get("promoted", False)) and str(st.get("recommended_layout_engine", "shadow")) == "v2":
                    resolved_layout_engine = "v2"
                else:
                    resolved_layout_engine = "shadow"
            except (json.JSONDecodeError, KeyError, TypeError):
                resolved_layout_engine = "shadow"
        else:
            resolved_layout_engine = "shadow"

    if args.db_path:
        stats = extract_from_papers_dir_with_db(
            args.papers_dir,
            args.out_dir,
            args.db_path,
            min_text_chars=args.min_text_chars,
            two_column_mode=args.two_column_mode,
            ocr_enabled=resolved_ocr_enabled,
            ocr_timeout_sec=resolved_ocr_timeout_sec,
            ocr_min_chars_trigger=resolved_ocr_min_chars_trigger,
            ocr_max_pages=resolved_ocr_max_pages,
            ocr_min_output_chars=resolved_ocr_min_output_chars,
            ocr_min_gain_chars=resolved_ocr_min_gain_chars,
            ocr_min_confidence=resolved_ocr_min_confidence,
            ocr_lang=resolved_ocr_lang,
            ocr_profile=resolved_ocr_profile,
            ocr_noise_suppression=resolved_ocr_noise_suppression,
            layout_engine=resolved_layout_engine,
            layout_table_handling=args.layout_table_handling,
            layout_footnote_handling=args.layout_footnote_handling,
            layout_min_region_confidence=args.layout_min_region_confidence,
        )
    else:
        stats = extract_from_papers_dir(
            args.papers_dir,
            args.out_dir,
            min_text_chars=args.min_text_chars,
            two_column_mode=args.two_column_mode,
            ocr_enabled=resolved_ocr_enabled,
            ocr_timeout_sec=resolved_ocr_timeout_sec,
            ocr_min_chars_trigger=resolved_ocr_min_chars_trigger,
            ocr_max_pages=resolved_ocr_max_pages,
            ocr_min_output_chars=resolved_ocr_min_output_chars,
            ocr_min_gain_chars=resolved_ocr_min_gain_chars,
            ocr_min_confidence=resolved_ocr_min_confidence,
            ocr_lang=resolved_ocr_lang,
            ocr_profile=resolved_ocr_profile,
            ocr_noise_suppression=resolved_ocr_noise_suppression,
            layout_engine=resolved_layout_engine,
            layout_table_handling=args.layout_table_handling,
            layout_footnote_handling=args.layout_footnote_handling,
            layout_min_region_confidence=args.layout_min_region_confidence,
        )
    print(json.dumps(stats, indent=2))


def cmd_layout_promotion_gate(args: argparse.Namespace) -> None:
    out = run_layout_promotion_gate(
        papers_dir=args.papers_dir,
        gold_path=args.gold,
        state_path=args.state_path,
        db_path=args.db_path,
        min_text_chars=args.min_text_chars,
        two_column_mode=args.two_column_mode,
        min_weighted_score=args.min_weighted_score,
        max_weighted_regression=args.max_weighted_regression,
        max_ordered_regression=args.max_ordered_regression,
        max_page_nonempty_regression=args.max_page_nonempty_regression,
    )
    print(json.dumps(out, indent=2))


def cmd_research(args: argparse.Namespace) -> None:
    out = run_research(
        args.index,
        args.question,
        args.top_k,
        args.out,
        min_items=args.min_items,
        min_score=args.min_score,
        use_cache=not args.no_cache,
        cache_dir=args.cache_dir,
        retrieval_mode=args.retrieval_mode,
        alpha=args.alpha,
        max_per_paper=args.max_per_paper,
        vector_index_path=args.vector_index_path,
        vector_metadata_path=args.vector_metadata_path,
        embedding_model=args.embedding_model,
        quality_prior_weight=args.quality_prior_weight,
        sources_config_path=args.sources_config,
        semantic_mode=args.semantic_mode,
        semantic_model=args.semantic_model,
        semantic_min_support=args.semantic_min_support,
        semantic_max_contradiction=args.semantic_max_contradiction,
        semantic_shadow_mode=args.semantic_shadow_mode,
        semantic_fail_on_low_support=args.semantic_fail_on_low_support,
        online_semantic_model=args.online_semantic_model,
        online_semantic_timeout_sec=args.online_semantic_timeout_sec,
        online_semantic_max_checks=args.online_semantic_max_checks,
        online_semantic_on_warn_only=args.online_semantic_on_warn_only,
        online_semantic_base_url=args.online_semantic_base_url,
        online_semantic_api_key=args.online_semantic_api_key,
        vector_service_endpoint=args.vector_service_endpoint,
        vector_nprobe=args.vector_nprobe,
        vector_ef_search=args.vector_ef_search,
        vector_topk_candidate_multiplier=args.vector_topk_candidate_multiplier,
        live_enabled=bool(args.live),
        live_sources_override=args.live_sources,
        live_max_items=args.live_max_items,
        live_timeout_sec=args.live_timeout_sec,
        live_cache_ttl_sec=args.live_cache_ttl_sec,
        live_merge_mode=args.live_merge_mode,
        routing_mode=args.routing_mode,
        intent=args.intent,
        relevance_policy=args.relevance_policy,
        diagnostics=args.diagnostics,
        direct_answer_mode=args.direct_answer_mode,
        direct_answer_max_complexity=args.direct_answer_max_complexity,
        use_kb=bool(args.use_kb),
        kb_db=args.kb_db,
        kb_top_k=args.kb_top_k,
        kb_merge_weight=args.kb_merge_weight,
        aggregate=bool(args.aggregate),
        aggregate_model=args.aggregate_model,
        aggregate_similarity_threshold=args.aggregate_similarity_threshold,
        aggregate_contradiction_threshold=args.aggregate_contradiction_threshold,
    )
    print(out)

    # Optional session recording
    session_id = getattr(args, "session_id", None)
    if session_id is not None:
        session_db = getattr(args, "session_db", DEFAULT_SESSION_DB)
        report_json_path = Path(out).with_suffix(".json")
        metrics_path = Path(out).with_suffix(".metrics.json")
        key_claims: list[str] = []
        gaps: list[str] = []
        citations: list[str] = []
        synthesis_preview = ""
        evidence_count = 0
        elapsed_ms: float | None = None
        aggregation_enabled = bool(args.aggregate)
        if report_json_path.exists():
            rdata = json.loads(report_json_path.read_text(encoding="utf-8"))
            key_claims = rdata.get("key_claims", [])
            gaps = rdata.get("gaps", [])
            citations = rdata.get("citations", [])
            synthesis_preview = rdata.get("synthesis", "")[:500]
        if metrics_path.exists():
            mdata = json.loads(metrics_path.read_text(encoding="utf-8"))
            elapsed_ms = mdata.get("elapsed_ms_total")
        evidence_path = Path(out).with_suffix(".evidence.json")
        if evidence_path.exists():
            edata = json.loads(evidence_path.read_text(encoding="utf-8"))
            evidence_count = len(edata.get("items", []))
        record_query(
            session_db,
            int(session_id),
            question=args.question,
            report_path=out,
            key_claims=key_claims,
            gaps=gaps,
            citations=citations,
            synthesis_preview=synthesis_preview,
            evidence_count=evidence_count,
            elapsed_ms=elapsed_ms,
            aggregation_enabled=aggregation_enabled,
        )
        print(f"Recorded in session {session_id}")


def _parse_params(params: list[str] | None) -> dict[str, str]:
    out: dict[str, str] = {}
    for raw in params or []:
        if "=" not in raw:
            raise SystemExit(f"Invalid --params value '{raw}'. Expected key=value.")
        k, v = raw.split("=", 1)
        k = k.strip()
        if not k:
            raise SystemExit(f"Invalid --params value '{raw}'. Expected key=value.")
        out[k] = v.strip()
    return out


# ---------------------------------------------------------------------------
# Source reliability commands
# ---------------------------------------------------------------------------


def cmd_reliability_list(args: argparse.Namespace) -> None:
    sources = list_source_reliability(args.reliability_db, sort_by=args.sort_by)
    if args.format == "json":
        print(json.dumps([{"source_name": s.source_name, "reliability_score": round(s.reliability_score, 4),
                           "total_events": s.total_events, "fetch_ok": s.fetch_success_count,
                           "fetch_err": s.fetch_error_count} for s in sources], indent=2))
    else:
        if not sources:
            print("No sources tracked yet.")
            return
        print(f"{'Source':<25} {'Score':>7} {'Events':>7} {'Fetch OK':>9} {'Fetch Err':>10}")
        print("-" * 62)
        for s in sources:
            print(f"{s.source_name:<25} {s.reliability_score:>7.4f} {s.total_events:>7} "
                  f"{s.fetch_success_count:>9} {s.fetch_error_count:>10}")


def cmd_reliability_show(args: argparse.Namespace) -> None:
    from dataclasses import asdict
    sr = get_reliability(args.reliability_db, args.source)
    events = get_feedback_events(args.reliability_db, args.source, limit=args.events)
    if args.format == "json":
        print(json.dumps({
            "source": asdict(sr),
            "recent_events": [{
                "event_id": e.event_id, "event_type": e.event_type,
                "value": e.value, "run_id": e.run_id, "created_at": e.created_at,
            } for e in events],
        }, indent=2))
    else:
        print(f"Source: {sr.source_name}")
        print(f"Reliability score: {sr.reliability_score:.4f}")
        print(f"Total events: {sr.total_events}")
        print(f"Fetch success/error: {sr.fetch_success_count}/{sr.fetch_error_count}")
        aq = f"{sr.avg_extraction_quality:.3f}" if sr.avg_extraction_quality is not None else "N/A"
        print(f"Avg extraction quality: {aq}")
        print(f"Claims confirmed/disputed/stale: {sr.claims_confirmed}/{sr.claims_disputed}/{sr.claims_stale}")
        print(f"User upvotes/downvotes: {sr.user_upvotes}/{sr.user_downvotes}")
        if events:
            print(f"\nRecent events (last {len(events)}):")
            for e in events:
                print(f"  [{e.created_at}] {e.event_type} = {e.value}")


def cmd_reliability_record(args: argparse.Namespace) -> None:
    eid = record_feedback(args.reliability_db, args.source, args.event, value=args.value, run_id=args.run_id)
    print(json.dumps({"event_id": eid, "source": args.source, "event_type": args.event, "value": args.value}, indent=2))


def cmd_reliability_recompute(args: argparse.Namespace) -> None:
    if args.source:
        score = recompute_reliability(args.reliability_db, args.source)
        print(json.dumps({"source": args.source, "reliability_score": round(score, 4)}, indent=2))
    else:
        scores = recompute_all(args.reliability_db)
        print(json.dumps({name: round(s, 4) for name, s in scores.items()}, indent=2))


def cmd_reliability_report(args: argparse.Namespace) -> None:
    from dataclasses import asdict
    report = reliability_report(args.reliability_db)
    if args.format == "json":
        out = json.dumps(asdict(report), indent=2)
    else:
        out = render_reliability_markdown(report)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(out, encoding="utf-8")
        print(f"Report written to {args.out}")
    else:
        print(out)


def cmd_reliability_delete(args: argparse.Namespace) -> None:
    delete_source(args.reliability_db, args.source)
    print(json.dumps({"deleted": True, "source": args.source}, indent=2))


# ---------------------------------------------------------------------------
# Session management commands
# ---------------------------------------------------------------------------


def cmd_session_create(args: argparse.Namespace) -> None:
    sid = create_session(args.session_db, args.name, description=args.description or "")
    print(json.dumps({"session_id": sid, "name": args.name, "status": "open"}, indent=2))


def cmd_session_list(args: argparse.Namespace) -> None:
    status_filter = args.status if args.status != "all" else None
    sessions = list_sessions(args.session_db, status=status_filter)
    rows = [
        {
            "session_id": s.session_id,
            "name": s.name,
            "status": s.status,
            "queries": s.query_count,
            "created_at": s.created_at,
        }
        for s in sessions
    ]
    print(json.dumps(rows, indent=2))


def cmd_session_show(args: argparse.Namespace) -> None:
    detail = get_session_detail(args.session_db, args.session_id)
    if args.format == "json":
        payload = {
            "session": {
                "session_id": detail.info.session_id,
                "name": detail.info.name,
                "description": detail.info.description,
                "status": detail.info.status,
                "created_at": detail.info.created_at,
                "closed_at": detail.info.closed_at,
                "query_count": detail.info.query_count,
            },
            "queries": [
                {
                    "query_id": q.query_id,
                    "question": q.question,
                    "report_path": q.report_path,
                    "created_at": q.created_at,
                    "elapsed_ms": q.elapsed_ms,
                    "key_claims": q.key_claims,
                    "gaps": q.gaps,
                    "citations": q.citations,
                    "evidence_count": q.evidence_count,
                    "aggregation_enabled": q.aggregation_enabled,
                }
                for q in detail.queries
            ],
        }
        print(json.dumps(payload, indent=2))
    else:
        print(render_session_markdown(detail))


def cmd_session_summary(args: argparse.Namespace) -> None:
    detail = get_session_detail(args.session_db, args.session_id)
    summary = session_summary(args.session_db, args.session_id)
    if args.format == "json":
        payload = {
            "session_id": summary.session_id,
            "session_name": summary.session_name,
            "query_count": summary.query_count,
            "total_evidence_items": summary.total_evidence_items,
            "unique_papers": summary.unique_papers,
            "total_citations": summary.total_citations,
            "unique_citations": summary.unique_citations,
            "all_gaps": summary.all_gaps,
            "recurring_gaps": summary.recurring_gaps,
            "all_key_claims": summary.all_key_claims,
            "question_timeline": summary.question_timeline,
            "avg_elapsed_ms": summary.avg_elapsed_ms,
            "aggregation_used_count": summary.aggregation_used_count,
            "coverage": summary.coverage,
        }
        print(json.dumps(payload, indent=2))
    else:
        md = render_session_markdown(detail, summary=summary)
        print(md)
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            render_session_markdown(detail, summary=summary),
            encoding="utf-8",
        )
        print(f"\nSaved to {args.out}")


def cmd_session_close(args: argparse.Namespace) -> None:
    close_session(args.session_db, args.session_id)
    print(json.dumps({"session_id": args.session_id, "status": "closed"}, indent=2))


def cmd_session_reopen(args: argparse.Namespace) -> None:
    reopen_session(args.session_db, args.session_id)
    print(json.dumps({"session_id": args.session_id, "status": "open"}, indent=2))


def cmd_session_delete(args: argparse.Namespace) -> None:
    delete_session(args.session_db, args.session_id)
    print(json.dumps({"session_id": args.session_id, "deleted": True}, indent=2))


def cmd_live_fetch(args: argparse.Namespace) -> None:
    cfg = load_sources_config(args.sources_config)
    live_cfg = live_sources_config(cfg)
    if args.source not in live_cfg:
        raise SystemExit(f"Live source not found in config: {args.source}")
    source = live_cfg[args.source]
    if not source.enabled:
        raise SystemExit(f"Live source is disabled: {args.source}")
    snap_cfg = live_snapshot_config(cfg)
    params = _parse_params(args.params)
    timeout_sec = int(args.live_timeout_sec or source.timeout_sec)
    max_items = int(args.live_max_items or source.limit or 20)
    max_body_bytes = int(snap_cfg.get("max_body_bytes", 2_000_000))
    ttl_sec = int(args.live_cache_ttl_sec) if args.live_cache_ttl_sec is not None else int(source.cache_ttl_sec)

    cached = load_cached_snapshot(
        root_dir=str(snap_cfg.get("root_dir")),
        source=source.name,
        query=args.query,
        params=params,
        ttl_sec=ttl_sec,
    )
    cache_hit = cached is not None
    if cached is not None:
        snap = cached
    else:
        items, raw_payload = fetch_live_source(
            source,
            query=args.query,
            params=params,
            timeout_sec=timeout_sec,
            max_items=max_items,
            max_body_bytes=max_body_bytes,
        )
        snap, _ = save_snapshot(
            root_dir=str(snap_cfg.get("root_dir")),
            source=source.name,
            query=args.query,
            params=params,
            items=items,
            persist_raw=bool(snap_cfg.get("persist_raw", True)),
            raw_payload=raw_payload,
        )

    snippets = snapshot_to_snippets(snap, max_items=max_items)
    default_name = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + ".json"
    out_path = Path(args.out) if args.out else Path("runs/live") / source.name / default_name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "source": source.name,
        "query": args.query,
        "cache_hit": cache_hit,
        "snapshot_id": snap.snapshot_id,
        "snapshot_path": str(Path(str(snap_cfg.get("root_dir"))) / source.name / f"{snap.cache_key}.snapshot.json"),
        "snippets": [s.model_dump() for s in snippets],
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(str(out_path))


def cmd_kb_ingest(args: argparse.Namespace) -> None:
    stats = ingest_report_to_kb(
        report_path=args.report,
        evidence_path=args.evidence,
        metrics_path=args.metrics,
        kb_db=args.kb_db,
        topic=args.topic,
        run_id=args.run_id,
    )
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))


def cmd_kb_query(args: argparse.Namespace) -> None:
    payload = run_kb_query(
        kb_db=args.kb_db,
        query=args.query,
        topic=args.topic,
        top_k=args.top_k,
    )
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


def cmd_kb_diff(args: argparse.Namespace) -> None:
    payload = run_kb_diff(
        kb_db=args.kb_db,
        topic=args.topic,
        since_run=args.since_run,
    )
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


def cmd_kb_contradiction_resolve(args: argparse.Namespace) -> None:
    payload = run_kb_contradiction_resolver(
        kb_db=args.kb_db,
        topic=args.topic,
        index_path=args.index,
        out_dir=args.out_dir,
        run_id=args.run_id,
        max_pairs=args.max_pairs,
        support_margin=args.support_margin,
        top_k=args.top_k,
        min_items=args.min_items,
        min_score=args.min_score,
        retrieval_mode=args.retrieval_mode,
        alpha=args.alpha,
        max_per_paper=args.max_per_paper,
        quality_prior_weight=args.quality_prior_weight,
        embedding_model=args.embedding_model,
        sources_config_path=args.sources_config,
        papers_db=args.papers_db,
        reliability_db=args.reliability_db,
    )
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


def cmd_kb_backfill(args: argparse.Namespace) -> None:
    payload = backfill_kb(
        kb_db=args.kb_db,
        reports_dir=args.reports_dir,
        topic=args.topic,
        last_n=args.last_n,
    )
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


def cmd_benchmark_research(args: argparse.Namespace) -> None:
    if args.queries_file:
        queries = [ln.strip() for ln in Path(args.queries_file).read_text(encoding="utf-8").splitlines() if ln.strip()]
    else:
        queries = args.query or []
    out = benchmark_research(
        index_path=args.index,
        queries=queries,
        runs_per_query=args.runs_per_query,
        top_k=args.top_k,
        min_items=args.min_items,
        min_score=args.min_score,
        retrieval_mode=args.retrieval_mode,
        alpha=args.alpha,
        max_per_paper=args.max_per_paper,
        vector_index_path=args.vector_index_path,
        vector_metadata_path=args.vector_metadata_path,
        embedding_model=args.embedding_model,
        quality_prior_weight=args.quality_prior_weight,
        sources_config_path=args.sources_config,
        semantic_mode=args.semantic_mode,
        semantic_model=args.semantic_model,
        semantic_min_support=args.semantic_min_support,
        semantic_max_contradiction=args.semantic_max_contradiction,
        semantic_shadow_mode=args.semantic_shadow_mode,
        semantic_fail_on_low_support=args.semantic_fail_on_low_support,
        online_semantic_model=args.online_semantic_model,
        online_semantic_timeout_sec=args.online_semantic_timeout_sec,
        online_semantic_max_checks=args.online_semantic_max_checks,
        online_semantic_on_warn_only=args.online_semantic_on_warn_only,
        online_semantic_base_url=args.online_semantic_base_url,
        online_semantic_api_key=args.online_semantic_api_key,
        vector_service_endpoint=args.vector_service_endpoint,
        vector_nprobe=args.vector_nprobe,
        vector_ef_search=args.vector_ef_search,
        vector_topk_candidate_multiplier=args.vector_topk_candidate_multiplier,
        out_path=args.out,
    )
    print(out)


def cmd_vector_service_build(args: argparse.Namespace) -> None:
    out = vector_service_build(
        index_path=args.index,
        vector_index_path=args.vector_index_path,
        vector_metadata_path=args.vector_metadata_path,
        embedding_model=args.embedding_model,
        batch_size=args.batch_size,
        vector_index_type=args.vector_index_type,
        vector_nlist=args.vector_nlist,
        vector_m=args.vector_m,
        vector_ef_construction=args.vector_ef_construction,
        vector_shards=args.vector_shards,
        vector_train_sample_size=args.vector_train_sample_size,
    )
    print(json.dumps(out, indent=2))


def cmd_vector_service_query(args: argparse.Namespace) -> None:
    out = vector_service_query(
        index_path=args.index,
        question=args.question,
        top_k=args.top_k,
        vector_index_path=args.vector_index_path,
        vector_metadata_path=args.vector_metadata_path,
        embedding_model=args.embedding_model,
        vector_nprobe=args.vector_nprobe,
        vector_ef_search=args.vector_ef_search,
    )
    print(json.dumps(out, indent=2))


def cmd_vector_service_health(args: argparse.Namespace) -> None:
    out = vector_service_health(
        index_path=args.index,
        vector_index_path=args.vector_index_path,
        vector_metadata_path=args.vector_metadata_path,
    )
    print(json.dumps(out, indent=2))


def cmd_vector_service_serve(args: argparse.Namespace) -> None:
    server, info = start_vector_service_server(
        index_path=args.index,
        vector_index_path=args.vector_index_path,
        vector_metadata_path=args.vector_metadata_path,
        host=args.host,
        port=args.port,
        embedding_model=args.embedding_model,
        vector_nprobe=args.vector_nprobe,
        vector_ef_search=args.vector_ef_search,
    )
    print(json.dumps(info, indent=2))
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def cmd_benchmark_scale(args: argparse.Namespace) -> None:
    if args.queries_file:
        queries = [ln.strip() for ln in Path(args.queries_file).read_text(encoding="utf-8").splitlines() if ln.strip()]
    else:
        queries = args.query or []
    out = benchmark_scale(
        corpus_dir=args.corpus,
        queries=queries,
        repeat_factor=args.repeat_factor,
        runs_per_query=args.runs_per_query,
        top_k=args.top_k,
        out_path=args.out,
    )
    print(out)


def cmd_corpus_health(args: argparse.Namespace) -> None:
    stats = evaluate_corpus_health(
        corpus_dir=args.corpus,
        extract_stats_path=args.extract_stats,
        min_snippets=args.min_snippets,
        min_avg_chars_per_paper=args.min_avg_chars_per_paper,
        min_avg_chars_per_page=args.min_avg_chars_per_page,
        max_extract_error_rate=args.max_extract_error_rate,
    )
    print(json.dumps(stats, indent=2))
    if args.fail_on_unhealthy and not bool(stats.get("healthy", False)):
        raise SystemExit("Corpus health check failed")


def cmd_extraction_quality_report(args: argparse.Namespace) -> None:
    out = write_extraction_quality_report(corpus_dir=args.corpus, out_path=args.out)
    print(out)


def cmd_extraction_eval(args: argparse.Namespace) -> None:
    out, report = write_extraction_eval_report(corpus_dir=args.corpus, gold_path=args.gold, out_path=args.out)
    print(out)
    print(json.dumps(report.get("summary", {}), indent=2))
    if args.fail_below is not None:
        score = float((report.get("summary", {}) or {}).get("weighted_score", 0.0))
        if score < float(args.fail_below):
            raise SystemExit(f"Extraction eval failed: weighted_score {score:.4f} < {float(args.fail_below):.4f}")


def cmd_extraction_eval_scaffold_gold(args: argparse.Namespace) -> None:
    out = scaffold_extraction_gold(
        corpus_dir=args.corpus,
        out_path=args.out,
        max_papers=args.max_papers,
        checks_per_paper=args.checks_per_paper,
        min_chars=args.min_chars,
    )
    print(out)


def cmd_run_pipeline(args: argparse.Namespace) -> None:
    summary = run_full_pipeline(
        query=args.query,
        db_path=args.db_path,
        papers_dir=args.papers_dir,
        extracted_dir=args.extracted_dir,
        index_path=args.index,
        report_path=args.report,
        sources_config_path=args.sources_config,
        max_results=args.max_results,
        with_semantic_scholar=args.with_semantic_scholar,
        top_k=args.top_k,
        min_items=args.min_items,
        min_score=args.min_score,
        min_text_chars=args.min_text_chars,
        min_snippets=args.min_snippets,
        min_avg_chars_per_paper=args.min_avg_chars_per_paper,
        min_avg_chars_per_page=args.min_avg_chars_per_page,
        max_extract_error_rate=args.max_extract_error_rate,
        require_healthy_corpus=args.require_healthy_corpus,
        fail_on_source_quality_gate=args.fail_on_source_quality_gate,
        extraction_gold_path=args.extraction_gold,
        extraction_eval_fail_below=args.extraction_eval_fail_below,
    )
    summary_out = Path(args.summary_out)
    summary_out.parent.mkdir(parents=True, exist_ok=True)
    summary_out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if args.dispatch_alerts:
        try:
            from src.apps.automation import dispatch_alerts, load_automation_config

            alert_cfg = load_automation_config(args.alerts_config)
            run_summary = {
                "run_id": summary.get("run_id", "pipeline_manual"),
                "created_utc": summary.get("created_utc"),
                "any_topic_failure": bool(summary.get("status") != "ok"),
                "sync": summary.get("stages", {}).get("sync", {}),
                "corpus_health": summary.get("stages", {}).get("corpus_health", {}),
                "topics": [
                    {
                        "name": "pipeline_query",
                        "report_path": summary.get("stages", {}).get("research", {}).get("report_path"),
                        "validate_ok": bool(summary.get("stages", {}).get("validate_report", {}).get("ok", False)),
                        "evidence_usage": (summary.get("stages", {}).get("validate_report", {}).get("metrics", {}) or {}).get(
                            "evidence_usage"
                        ),
                    }
                ],
            }
            alert_result = dispatch_alerts(run_summary, alert_cfg)
            summary["alert_dispatch"] = alert_result
            alert_out = Path(args.alert_out)
            alert_out.parent.mkdir(parents=True, exist_ok=True)
            alert_out.write_text(json.dumps(alert_result, indent=2), encoding="utf-8")
            summary_out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        except Exception as exc:
            summary["alert_dispatch"] = {"enabled": bool(args.dispatch_alerts), "sent": False, "error": str(exc)}
            summary_out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"summary_written:{summary_out}")
    if summary.get("status") != "ok":
        raise SystemExit(f"Pipeline failed at stage: {summary.get('failed_stage')}")


def cmd_run_automation(args: argparse.Namespace) -> None:
    run_dir = run_automation(args.config)
    print(run_dir)


def cmd_serve_ui(args: argparse.Namespace) -> None:
    try:
        import uvicorn
    except Exception as exc:
        raise SystemExit(f"uvicorn is required for serve-ui: {exc}") from exc

    from src.ui.app import create_app
    from src.ui.settings import load_ui_settings

    settings = load_ui_settings(args.config)
    host = args.host or settings.host
    port = int(args.port if args.port is not None else settings.port)
    app = create_app(args.config)
    uvicorn.run(app, host=host, port=port, reload=bool(args.reload))


def cmd_migrate_extraction_meta(args: argparse.Namespace) -> None:
    stats = migrate_extraction_meta(corpus_dir=args.corpus, dry_run=args.dry_run)
    print(json.dumps(stats, indent=2))


def cmd_snapshot_run(args: argparse.Namespace) -> None:
    out = create_snapshot(
        out_dir=args.out,
        report_path=args.report,
        index_path=args.index,
        config_paths=args.config or [],
    )
    print(out)


def cmd_doc_write(args: argparse.Namespace) -> None:
    out = write_doc(args.doc_type, args.facts, args.out)
    print(out)


def cmd_release_notes(args: argparse.Namespace) -> None:
    out = generate_release_notes(args.from_ref, args.to_ref, args.out)
    print(out)


def cmd_train_tinygpt(args: argparse.Namespace) -> None:
    from src.train import train

    ckpt = train(data_path=args.data, out_dir=args.out_dir, config_path=args.config)
    print(ckpt)


def cmd_generate(args: argparse.Namespace) -> None:
    from src.generate import generate_text

    text = generate_text(
        checkpoint=args.checkpoint,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        deterministic=args.deterministic,
        seed=args.seed,
    )
    print(text)


def cmd_validate_report(args: argparse.Namespace) -> None:
    payload = json.loads(Path(args.evidence).read_text(encoding="utf-8"))
    evidence = EvidencePack(**payload)

    # Optional structural check if companion JSON report exists.
    report_json = Path(args.input).with_suffix(".json")
    errors = []
    is_direct_answer = False
    if report_json.exists():
        from src.core.schemas import ResearchReport

        rep = ResearchReport(**json.loads(report_json.read_text(encoding="utf-8")))
        is_direct_answer = bool((rep.retrieval_diagnostics or {}).get("direct_answer_used", False))
        errors.extend(validate_report_citations(rep))
        if not is_direct_answer:
            errors.extend(validate_claim_support(rep, evidence))
            sem_errors, sem_metrics, sem_warnings = validate_semantic_claim_support(
                rep,
                evidence,
                semantic_mode=args.semantic_mode,
                model_name=args.semantic_model,
                min_support=args.semantic_min_support,
                max_contradiction=args.semantic_max_contradiction,
                shadow_mode=args.semantic_shadow_mode,
                fail_on_low_support=args.semantic_fail_on_low_support,
                online_model=args.online_semantic_model,
                online_timeout_sec=args.online_semantic_timeout_sec,
                online_max_checks=args.online_semantic_max_checks,
                online_on_warn_only=args.online_semantic_on_warn_only,
                online_base_url=args.online_semantic_base_url,
                online_api_key=args.online_semantic_api_key,
            ) if args.semantic_faithfulness else ([], {"semantic_checked": False, "semantic_disabled": True}, [])
            errors.extend(sem_errors)
        else:
            sem_metrics = {"semantic_checked": False, "semantic_skipped": "direct_answer"}
            sem_warnings = []
        report_text = rep.synthesis + "\n" + "\n".join(rep.gaps + [e.proposal for e in rep.experiments])
        metrics = report_coverage_metrics(rep, evidence)
        metrics.update(sem_metrics)
        for w in sem_warnings:
            log.warning("%s", w)
    else:
        report_text = Path(args.input).read_text(encoding="utf-8")
        metrics = {"semantic_checked": False, "semantic_skipped": "report_json_missing"} if args.semantic_faithfulness else None

    # Direct-answer reports are deterministic computations and intentionally have no evidence pack.
    if not is_direct_answer:
        errors.extend(validate_no_new_numbers(report_text, evidence))
    if errors:
        raise SystemExit("Validation failed:\n- " + "\n- ".join(errors))
    print("OK")
    if metrics is not None:
        print(json.dumps(metrics, indent=2))


def cmd_export_report(args: argparse.Namespace) -> None:
    report = load_report_json(args.input)
    evidence = load_evidence_json(args.evidence) if args.evidence else None
    db_path = args.papers_db if args.papers_db else None
    formats = [f.strip() for f in args.format.split(",")]
    if len(formats) == 1:
        fmt = formats[0]
        if args.out:
            path = export_report_to_file(
                report, args.out, evidence, fmt=fmt, papers_db_path=db_path,
            )
            print(f"Wrote {fmt} export to {path}")
        else:
            print(export_report(report, evidence, fmt=fmt, papers_db_path=db_path))
    else:
        out_dir = args.out or "runs/research_reports"
        basename = Path(args.input).stem
        results = export_report_multi(
            report, out_dir, basename=basename, evidence=evidence,
            formats=formats, papers_db_path=db_path,
        )
        for fmt, path in results.items():
            print(f"Wrote {fmt} export to {path}")


def cmd_scaffold_domain_config(args: argparse.Namespace) -> None:
    out = scaffold_domain_config(
        domain=args.domain,
        out_path=args.out,
        profile=args.profile,
        overwrite=args.overwrite,
    )
    print(out)


def _slug(text: str) -> str:
    out = "".join(ch.lower() if ch.isalnum() else "_" for ch in text.strip())
    out = "_".join(part for part in out.split("_") if part)
    return out or "query"


def _default_query_out(mode: str, question: str) -> Path:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    slug = _slug(question)[:80]
    return Path("runs/query") / ts / f"{mode}_{slug}.md"


def cmd_ask(args: argparse.Namespace) -> None:
    router_cfg = load_router_config(args.config)
    out = args.out or str(_default_query_out("ask", args.question))
    payload = run_ask_mode(
        question=args.question,
        out_path=out,
        router_cfg=router_cfg,
        glossary_path=args.glossary,
    )
    print(payload["out_path"])


def cmd_monitor(args: argparse.Namespace) -> None:
    router_cfg = load_router_config(args.config)
    schedule = args.schedule or str(
        (((router_cfg.get("router", {}) or {}).get("monitor", {}) or {}).get("default_schedule_cron", "0 */6 * * *"))
    )
    slug = _slug(args.question)[:80]
    out = args.out or str(Path("runs/monitor") / f"{slug}.json")
    payload = run_monitor_mode(
        question=args.question,
        schedule=schedule,
        automation_config_path=args.automation_config,
        out_path=out,
        register_schedule=bool(args.register_schedule),
        schedule_backend=str(args.schedule_backend),
    )
    print(json.dumps(payload, indent=2))


def cmd_monitor_unregister(args: argparse.Namespace) -> None:
    token = args.name or args.question
    if not token:
        raise SystemExit("monitor-unregister requires --name or --question")
    payload = unregister_monitor(
        name_or_question=str(token),
        delete_files=bool(args.delete_files),
    )
    print(json.dumps(payload, indent=2))


def cmd_notes(args: argparse.Namespace) -> None:
    slug = _slug(args.topic or args.question)[:80]
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = args.out or str(Path("runs/notes") / f"{ts}_{slug}.md")
    payload = run_notes_mode(
        question=args.question,
        index_path=args.index,
        kb_db=args.kb_db,
        topic=args.topic,
        out_path=out,
        sources_config_path=args.sources_config,
    )
    print(payload["notes_path"])


def cmd_query(args: argparse.Namespace) -> None:
    router_cfg = load_router_config(args.config)
    dispatch_started_utc = datetime.now(timezone.utc).isoformat()
    t0 = time.perf_counter()
    decision = route_query(args.question, router_cfg, explicit_mode=args.mode)
    out = Path(args.out) if args.out else _default_query_out(decision.mode, args.question)
    out.parent.mkdir(parents=True, exist_ok=True)

    def _run_ask() -> dict:
        payload = run_ask_mode(
            question=args.question,
            out_path=str(out),
            router_cfg=router_cfg,
            glossary_path=args.glossary,
        )
        return {"output_path": payload["out_path"], "details": payload}

    def _run_research() -> dict:
        result = run_research(
            args.index,
            args.question,
            args.top_k,
            str(out),
            min_items=args.min_items,
            min_score=args.min_score,
            retrieval_mode=args.retrieval_mode,
            alpha=args.alpha,
            max_per_paper=args.max_per_paper,
            vector_index_path=args.vector_index_path,
            vector_metadata_path=args.vector_metadata_path,
            embedding_model=args.embedding_model,
            quality_prior_weight=args.quality_prior_weight,
            sources_config_path=args.sources_config,
            semantic_mode=args.semantic_mode,
            semantic_model=args.semantic_model,
            semantic_min_support=args.semantic_min_support,
            semantic_max_contradiction=args.semantic_max_contradiction,
            semantic_shadow_mode=args.semantic_shadow_mode,
            semantic_fail_on_low_support=args.semantic_fail_on_low_support,
            online_semantic_model=args.online_semantic_model,
            online_semantic_timeout_sec=args.online_semantic_timeout_sec,
            online_semantic_max_checks=args.online_semantic_max_checks,
            online_semantic_on_warn_only=args.online_semantic_on_warn_only,
            online_semantic_base_url=args.online_semantic_base_url,
            online_semantic_api_key=args.online_semantic_api_key,
            vector_service_endpoint=args.vector_service_endpoint,
            vector_nprobe=args.vector_nprobe,
            vector_ef_search=args.vector_ef_search,
            vector_topk_candidate_multiplier=args.vector_topk_candidate_multiplier,
            live_enabled=bool(args.live),
            live_sources_override=args.live_sources,
            live_max_items=args.live_max_items,
            live_timeout_sec=args.live_timeout_sec,
            live_cache_ttl_sec=args.live_cache_ttl_sec,
            live_merge_mode=args.live_merge_mode,
            routing_mode=args.routing_mode,
            intent=args.intent,
            relevance_policy=args.relevance_policy,
            diagnostics=args.diagnostics,
            direct_answer_mode=args.direct_answer_mode,
            direct_answer_max_complexity=args.direct_answer_max_complexity,
            use_kb=bool(args.use_kb),
            kb_db=args.kb_db,
            kb_top_k=args.kb_top_k,
            kb_merge_weight=args.kb_merge_weight,
        )
        return {"output_path": result, "details": {"mode": "research"}}

    def _run_monitor() -> dict:
        schedule = args.schedule or str(
            (((router_cfg.get("router", {}) or {}).get("monitor", {}) or {}).get("default_schedule_cron", "0 */6 * * *"))
        )
        monitor_json = out.with_suffix(".monitor.json")
        payload = run_monitor_mode(
            question=args.question,
            schedule=schedule,
            automation_config_path=args.automation_config,
            out_path=str(monitor_json),
            register_schedule=bool(args.register_schedule),
            schedule_backend=str(args.schedule_backend),
        )
        md = [
            "# Monitor",
            "",
            f"- question: {args.question}",
            f"- query: {payload.get('query')}",
            f"- schedule: {payload.get('schedule')}",
            f"- schedule_register_requested: {payload.get('schedule_register_requested')}",
            f"- schedule_backend_requested: {payload.get('schedule_backend_requested')}",
            f"- schedule_backend_used: {payload.get('schedule_backend_used')}",
            f"- schedule_registered: {payload.get('schedule_registered')}",
            f"- monitor_spec_path: {payload.get('monitor_spec_path')}",
            f"- generated_automation_config: {payload.get('generated_automation_config')}",
            f"- monitor_bootstrap_ok: {payload.get('monitor_bootstrap_ok')}",
            f"- baseline_run_id: {payload.get('baseline_run_id')}",
            f"- baseline_run_dir: {payload.get('baseline_run_dir')}",
        ]
        if payload.get("schedule_error"):
            md.append(f"- schedule_error: {payload.get('schedule_error')}")
        if payload.get("monitor_bootstrap_error"):
            md.append(f"- monitor_bootstrap_error: {payload.get('monitor_bootstrap_error')}")
        out.write_text("\n".join(md) + "\n", encoding="utf-8")
        return {"output_path": str(out), "details": payload}

    def _run_notes() -> dict:
        payload = run_notes_mode(
            question=args.question,
            index_path=args.index,
            kb_db=args.kb_db,
            topic=args.topic,
            out_path=str(out),
            sources_config_path=args.sources_config,
        )
        return {"output_path": payload["notes_path"], "details": payload}

    dispatch = dispatch_query(
        decision,
        {
            "ask": _run_ask,
            "research": _run_research,
            "monitor": _run_monitor,
            "notes": _run_notes,
        },
    )
    elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 3)
    router_sidecar = out.with_suffix(".router.json")
    router_sidecar.write_text(
        json.dumps(
            {
                "question": args.question,
                "mode_selected": decision.mode,
                "mode_confidence": decision.confidence,
                "mode_reason": decision.reason,
                "signals": decision.signals,
                "override_used": decision.override_used,
                "dispatch_started_utc": dispatch_started_utc,
                "dispatch_elapsed_ms": elapsed_ms,
                "output_path": dispatch["output_path"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(dispatch["output_path"])


def cmd_monitor_evaluate(args: argparse.Namespace) -> None:
    cfg = load_automation_config(args.config)
    run_dir = Path(args.run_dir)
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        raise SystemExit(f"manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    run_id = str(manifest.get("run_id") or run_dir.name)
    topic_state = next((t for t in (manifest.get("topics", []) or []) if str(t.get("name")) == args.topic), None)
    if topic_state is None:
        raise SystemExit(f"topic not found in manifest: {args.topic}")
    topic_cfg = next((t for t in (cfg.get("topics", []) or []) if str(t.get("name")) == args.topic), {"name": args.topic})
    payload = evaluate_topic_monitoring(
        cfg=cfg,
        run_id=run_id,
        run_dir=str(run_dir),
        topic=topic_cfg,
        topic_state=dict(topic_state),
    )
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


def cmd_monitor_digest_flush(args: argparse.Namespace) -> None:
    cfg = load_automation_config(args.config)
    payload = monitor_digest_flush(config=cfg)
    if payload.get("events"):
        summary = {
            "run_id": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "any_topic_failure": False,
            "sync": {"source_errors": 0},
            "corpus_health": {"healthy": True, "reasons": [], "warnings": []},
            "topics": [],
            "monitoring": {"events": payload.get("events", [])},
        }
        payload["dispatch"] = dispatch_alerts(summary, cfg)
    print(json.dumps(payload, indent=2))


def cmd_monitor_status(args: argparse.Namespace) -> None:
    cfg = load_automation_config(args.config)
    payload = monitor_status(config=cfg)
    print(json.dumps(payload, indent=2))


def cmd_monitor_soak_sim(args: argparse.Namespace) -> None:
    payload = run_monitor_soak_sim(
        topic=args.topic,
        runs=args.runs,
        interval_minutes=args.interval_minutes,
        cooldown_minutes=args.cooldown_minutes,
        hysteresis_runs=args.hysteresis_runs,
        pattern=args.pattern,
        trigger_every=args.trigger_every,
        burst_len=args.burst_len,
        gap_len=args.gap_len,
        severity=args.severity,
        metric=args.metric,
        op=args.op,
        threshold=args.threshold,
        bad_value=args.bad_value,
        good_value=args.good_value,
        out_path=args.out,
    )
    print(json.dumps(payload, indent=2))


def cmd_monitor_history_check(args: argparse.Namespace) -> None:
    payload = run_monitor_history_check(
        topic=args.topic,
        audit_dir=args.audit_dir,
        out_path=args.out,
    )
    print(json.dumps(payload, indent=2))


def cmd_query_router_eval(args: argparse.Namespace) -> None:
    payload = run_query_router_eval(
        cases_path=args.cases,
        router_config_path=args.config,
        out_path=args.out,
        strict_min_accuracy=args.strict_min_accuracy,
    )
    print(json.dumps(payload, indent=2))
    if bool(payload.get("gate_applied")) and not bool(payload.get("gate_passed", True)):
        got = float(payload.get("accuracy", 0.0))
        need = float(args.strict_min_accuracy)
        raise SystemExit(f"query-router-eval gate failed: accuracy={got:.4f} < strict_min_accuracy={need:.4f}")


def cmd_research_template(args: argparse.Namespace) -> None:
    payload = run_research_template(
        template_name=args.template,
        topic=args.topic,
        index_path=args.index,
        out_dir=args.out_dir,
        session_db=args.session_db,
        templates_path=args.templates_config,
        sources_config_path=args.sources_config,
        top_k=args.top_k,
        min_items=args.min_items,
        min_score=args.min_score,
        retrieval_mode=args.retrieval_mode,
        alpha=args.alpha,
        max_per_paper=args.max_per_paper,
        quality_prior_weight=args.quality_prior_weight,
        embedding_model=args.embedding_model,
    )
    print(json.dumps(payload, indent=2))


def cmd_watch_ingest(args: argparse.Namespace) -> None:
    payload = run_watch_ingest(
        watch_dir=args.dir,
        extracted_dir=args.extracted_dir,
        db_path=args.db_path,
        index_path=args.index,
        once=bool(args.once),
        poll_interval_sec=float(args.poll_interval_sec),
        max_events=int(args.max_events),
        min_text_chars=int(args.min_text_chars),
        out_path=args.out,
    )
    print(json.dumps(payload, indent=2))


def cmd_benchmark_regression_check(args: argparse.Namespace) -> None:
    payload = run_benchmark_regression_check(
        benchmark_path=args.benchmark,
        history_path=args.history,
        run_id=args.run_id,
        max_latency_regression_pct=args.max_latency_regression_pct,
        min_quality_floor=args.min_quality_floor,
        history_window=args.history_window,
    )
    if args.out:
        p = Path(args.out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    if bool(payload.get("regressed", False)) and bool(args.fail_on_regression):
        raise SystemExit("benchmark regression gate failed")


def cmd_reliability_watchdog(args: argparse.Namespace) -> None:
    cfg: dict[str, Any] = {"enabled": True}
    if args.config:
        acfg = load_automation_config(args.config)
        cfg = dict((acfg.get("reliability", {}) if isinstance(acfg, dict) else {}) or {})
        cfg["enabled"] = True
    if args.reliability_db:
        cfg["db_path"] = args.reliability_db
    if args.state_path:
        cfg["state_path"] = args.state_path
    if args.report_path:
        cfg["report_path"] = args.report_path
    if args.degrade_threshold is not None:
        cfg["degrade_threshold"] = float(args.degrade_threshold)
    if args.critical_threshold is not None:
        cfg["critical_threshold"] = float(args.critical_threshold)
    if args.auto_disable_after is not None:
        cfg["auto_disable_after"] = int(args.auto_disable_after)

    sync_stats: list[dict[str, Any]] = []
    if args.run_dir:
        run_dir = Path(args.run_dir)
        manifest_path = run_dir / "manifest.json"
        if not manifest_path.exists():
            raise SystemExit(f"manifest not found: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for topic in manifest.get("topics", []) or []:
            sp = topic.get("sync_stats_file")
            if sp and Path(sp).exists():
                sync_stats.append(json.loads(Path(sp).read_text(encoding="utf-8")))
    for sp in args.sync_stats or []:
        p = Path(sp)
        if not p.exists():
            raise SystemExit(f"sync stats file not found: {p}")
        sync_stats.append(json.loads(p.read_text(encoding="utf-8")))
    if not sync_stats:
        raise SystemExit("reliability-watchdog requires --run-dir or at least one --sync-stats file")

    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    payload = run_reliability_watchdog(sync_stats_list=sync_stats, reliability_cfg=cfg, run_id=run_id)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Offline Research Copilot + Tiny GPT")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("sync-papers", help='sync-papers --query "..." --max-results N')
    p.add_argument("--query", required=True)
    p.add_argument("--max-results", type=int, default=20)
    p.add_argument("--db-path", default="data/papers.db")
    p.add_argument("--papers-dir", default="data/papers")
    p.add_argument("--with-semantic-scholar", action="store_true")
    p.add_argument("--prefer-arxiv", action="store_true")
    p.add_argument("--require-pdf", action="store_true")
    p.add_argument(
        "--sources-config",
        default="config/sources.yaml",
        help="Sources YAML config path (default: config/sources.yaml)",
    )
    p.add_argument("--fail-on-source-quality-gate", action="store_true")
    p.set_defaults(func=cmd_sync_papers)

    p = sub.add_parser("build-index", help="build-index --corpus data/extracted")
    p.add_argument("--corpus", default="data/extracted")
    p.add_argument("--out", default="data/indexes/bm25_index.json")
    p.add_argument(
        "--incremental",
        action="store_true",
        help="Incrementally update the existing index (append new snippets, skip unchanged corpus)",
    )
    p.add_argument("--db-path", default=None, help="Optional SQLite DB path to enrich snippet metadata")
    p.add_argument("--with-vector", action="store_true")
    p.add_argument("--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2")
    p.add_argument("--vector-index-path", default=None)
    p.add_argument("--vector-metadata-path", default=None)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--vector-index-type", choices=["flat", "ivf_flat", "hnsw"], default="flat")
    p.add_argument("--vector-nlist", type=int, default=1024)
    p.add_argument("--vector-m", type=int, default=32)
    p.add_argument("--vector-ef-construction", type=int, default=200)
    p.add_argument("--vector-shards", type=int, default=1)
    p.add_argument("--vector-train-sample-size", type=int, default=200000)
    p.add_argument("--require-healthy-corpus", action="store_true")
    p.add_argument("--min-snippets", type=int, default=1)
    p.add_argument("--min-avg-chars-per-paper", type=int, default=500)
    p.add_argument("--min-avg-chars-per-page", type=int, default=80)
    p.add_argument("--max-extract-error-rate", type=float, default=0.8)
    p.add_argument(
        "--live-data",
        action="append",
        default=[],
        help="Optional path to normalized live snippets payload JSON; repeatable",
    )
    p.set_defaults(func=cmd_build_index)

    p = sub.add_parser("extract-corpus", help="extract-corpus --papers-dir data/papers --out-dir data/extracted")
    p.add_argument("--papers-dir", default="data/papers")
    p.add_argument("--out-dir", default="data/extracted")
    p.add_argument("--db-path", default=None, help="Optional DB path to map PDFs back to canonical paper metadata")
    p.add_argument(
        "--sources-config",
        default="config/sources.yaml",
        help="Sources YAML config path for domain OCR defaults (default: config/sources.yaml)",
    )
    p.add_argument("--min-text-chars", type=int, default=200)
    p.add_argument("--two-column-mode", choices=["off", "auto", "force"], default="auto")
    p.add_argument("--layout-engine", choices=["legacy", "v2", "shadow", "auto"], default="shadow")
    p.add_argument("--layout-promotion-state", default="runs/audit/layout_promotion_state.json")
    p.add_argument("--layout-table-handling", choices=["drop", "linearize", "preserve"], default="linearize")
    p.add_argument("--layout-footnote-handling", choices=["drop", "append", "preserve"], default="append")
    p.add_argument("--layout-min-region-confidence", type=float, default=0.55)
    p.add_argument("--ocr-enabled", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--ocr-timeout-sec", type=int, default=None)
    p.add_argument("--ocr-min-chars-trigger", type=int, default=None)
    p.add_argument("--ocr-max-pages", type=int, default=None)
    p.add_argument("--ocr-min-output-chars", type=int, default=None)
    p.add_argument("--ocr-min-gain-chars", type=int, default=None)
    p.add_argument("--ocr-min-confidence", type=float, default=None)
    p.add_argument("--ocr-lang", default=None, help='OCR language, e.g. "eng", "eng+spa", or "auto"')
    p.add_argument("--ocr-profile", choices=["document", "sparse"], default=None)
    p.add_argument("--ocr-noise-suppression", action=argparse.BooleanOptionalAction, default=None)
    p.set_defaults(func=cmd_extract_corpus)

    p = sub.add_parser("research", help="research --question ... --top-k K --out ...")
    p.add_argument("--index", default="data/indexes/bm25_index.json")
    p.add_argument("--question", required=True)
    p.add_argument("--top-k", type=int, default=8)
    p.add_argument("--min-items", type=int, default=2)
    p.add_argument("--min-score", type=float, default=0.5)
    p.add_argument("--retrieval-mode", choices=["lexical", "vector", "hybrid"], default="lexical")
    p.add_argument("--alpha", type=float, default=0.6)
    p.add_argument("--max-per-paper", type=int, default=2)
    p.add_argument("--vector-index-path", default=None)
    p.add_argument("--vector-metadata-path", default=None)
    p.add_argument("--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2")
    p.add_argument("--quality-prior-weight", type=float, default=0.15)
    p.add_argument("--vector-service-endpoint", default=None)
    p.add_argument("--vector-nprobe", type=int, default=16)
    p.add_argument("--vector-ef-search", type=int, default=64)
    p.add_argument("--vector-topk-candidate-multiplier", type=float, default=1.5)
    p.add_argument("--live", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--live-sources", default=None, help="Comma list of live sources to query")
    p.add_argument("--live-max-items", type=int, default=20)
    p.add_argument("--live-timeout-sec", type=int, default=20)
    p.add_argument("--live-cache-ttl-sec", type=int, default=None)
    p.add_argument("--live-merge-mode", choices=["union", "live_first"], default="union")
    p.add_argument("--routing-mode", choices=["auto", "manual"], default="auto")
    p.add_argument("--intent", default=None)
    p.add_argument("--relevance-policy", choices=["not_found", "warn", "fail"], default="not_found")
    p.add_argument("--diagnostics", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--direct-answer-mode", choices=["off", "hybrid"], default="hybrid")
    p.add_argument("--direct-answer-max-complexity", type=int, default=2)
    p.add_argument("--use-kb", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--kb-db", default="data/kb/knowledge.db")
    p.add_argument("--kb-top-k", type=int, default=5)
    p.add_argument("--kb-merge-weight", type=float, default=0.15)
    p.add_argument("--semantic-mode", choices=["offline", "online", "hybrid"], default="offline")
    p.add_argument("--semantic-model", default="sentence-transformers/all-MiniLM-L6-v2")
    p.add_argument("--semantic-min-support", type=float, default=0.55)
    p.add_argument("--semantic-max-contradiction", type=float, default=0.30)
    p.add_argument("--semantic-shadow-mode", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--semantic-fail-on-low-support", action="store_true")
    p.add_argument("--online-semantic-model", default="gpt-4o-mini")
    p.add_argument("--online-semantic-timeout-sec", type=float, default=12.0)
    p.add_argument("--online-semantic-max-checks", type=int, default=12)
    p.add_argument("--online-semantic-on-warn-only", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--online-semantic-base-url", default=None)
    p.add_argument("--online-semantic-api-key", default=None)
    p.add_argument(
        "--sources-config",
        default="config/sources.yaml",
        help="Sources YAML config for ranking/limits (default: config/sources.yaml)",
    )
    p.add_argument("--no-cache", action="store_true")
    p.add_argument("--cache-dir", default="runs/cache/research")
    p.add_argument("--out", default="runs/research_reports/report.md")
    p.add_argument("--aggregate", action=argparse.BooleanOptionalAction, default=False,
                   help="Enable cross-document evidence aggregation (clusters + consensus/conflict)")
    p.add_argument("--aggregate-model", default="sentence-transformers/all-MiniLM-L6-v2",
                   help="Embedding model for evidence clustering")
    p.add_argument("--aggregate-similarity-threshold", type=float, default=0.55,
                   help="Cosine similarity threshold for clustering (0-1)")
    p.add_argument("--aggregate-contradiction-threshold", type=float, default=0.40,
                   help="Contradiction score threshold for conflict labelling (0-1)")
    p.add_argument("--session-id", type=int, default=None,
                   help="Record this research query in the given session")
    p.add_argument("--session-db", default=DEFAULT_SESSION_DB,
                   help="Path to session tracking database")
    p.set_defaults(func=cmd_research)

    p = sub.add_parser("ask", help="quick deterministic answer mode")
    p.add_argument("--question", required=True)
    p.add_argument("--config", default="config/router.yaml")
    p.add_argument("--glossary", default="config/ask_glossary.yaml")
    p.add_argument("--out", default=None)
    p.set_defaults(func=cmd_ask)

    p = sub.add_parser("monitor", help="create or update monitor topic and run baseline")
    p.add_argument("--question", required=True)
    p.add_argument("--config", default="config/router.yaml")
    p.add_argument("--automation-config", default="config/automation.yaml")
    p.add_argument("--schedule", default=None)
    p.add_argument("--register-schedule", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--schedule-backend", choices=["auto", "crontab", "file"], default="auto")
    p.add_argument("--out", default=None)
    p.set_defaults(func=cmd_monitor)

    p = sub.add_parser("monitor-unregister", help="remove monitor cron registration and optional generated files")
    grp = p.add_mutually_exclusive_group(required=True)
    grp.add_argument("--name", default=None, help="Monitor slug/name")
    grp.add_argument("--question", default=None, help="Question/prompt used to derive monitor slug")
    p.add_argument("--delete-files", action=argparse.BooleanOptionalAction, default=False)
    p.set_defaults(func=cmd_monitor_unregister)

    p = sub.add_parser("notes", help="research + KB ingest + structured notes")
    p.add_argument("--question", required=True)
    p.add_argument("--index", default="data/indexes/bm25_index.json")
    p.add_argument("--kb-db", default="data/kb/knowledge.db")
    p.add_argument("--topic", default=None)
    p.add_argument("--sources-config", default="config/sources.yaml")
    p.add_argument("--out", default=None)
    p.set_defaults(func=cmd_notes)

    p = sub.add_parser("query", help="route prompt into ask/research/monitor/notes")
    p.add_argument("--question", required=True)
    p.add_argument("--mode", choices=["auto", "ask", "research", "monitor", "notes"], default="auto")
    p.add_argument("--config", default="config/router.yaml")
    p.add_argument("--glossary", default="config/ask_glossary.yaml")
    p.add_argument("--automation-config", default="config/automation.yaml")
    p.add_argument("--schedule", default=None)
    p.add_argument("--register-schedule", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--schedule-backend", choices=["auto", "crontab", "file"], default="auto")
    p.add_argument("--index", default="data/indexes/bm25_index.json")
    p.add_argument("--kb-db", default="data/kb/knowledge.db")
    p.add_argument("--topic", default=None)
    p.add_argument("--top-k", type=int, default=8)
    p.add_argument("--min-items", type=int, default=2)
    p.add_argument("--min-score", type=float, default=0.5)
    p.add_argument("--retrieval-mode", choices=["lexical", "vector", "hybrid"], default="lexical")
    p.add_argument("--alpha", type=float, default=0.6)
    p.add_argument("--max-per-paper", type=int, default=2)
    p.add_argument("--vector-index-path", default=None)
    p.add_argument("--vector-metadata-path", default=None)
    p.add_argument("--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2")
    p.add_argument("--quality-prior-weight", type=float, default=0.15)
    p.add_argument("--vector-service-endpoint", default=None)
    p.add_argument("--vector-nprobe", type=int, default=16)
    p.add_argument("--vector-ef-search", type=int, default=64)
    p.add_argument("--vector-topk-candidate-multiplier", type=float, default=1.5)
    p.add_argument("--live", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--live-sources", default=None)
    p.add_argument("--live-max-items", type=int, default=20)
    p.add_argument("--live-timeout-sec", type=int, default=20)
    p.add_argument("--live-cache-ttl-sec", type=int, default=None)
    p.add_argument("--live-merge-mode", choices=["union", "live_first"], default="union")
    p.add_argument("--routing-mode", choices=["auto", "manual"], default="auto")
    p.add_argument("--intent", default=None)
    p.add_argument("--relevance-policy", choices=["not_found", "warn", "fail"], default="not_found")
    p.add_argument("--diagnostics", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--direct-answer-mode", choices=["off", "hybrid"], default="hybrid")
    p.add_argument("--direct-answer-max-complexity", type=int, default=2)
    p.add_argument("--use-kb", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--kb-top-k", type=int, default=5)
    p.add_argument("--kb-merge-weight", type=float, default=0.15)
    p.add_argument("--semantic-mode", choices=["offline", "online", "hybrid"], default="offline")
    p.add_argument("--semantic-model", default="sentence-transformers/all-MiniLM-L6-v2")
    p.add_argument("--semantic-min-support", type=float, default=0.55)
    p.add_argument("--semantic-max-contradiction", type=float, default=0.30)
    p.add_argument("--semantic-shadow-mode", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--semantic-fail-on-low-support", action="store_true")
    p.add_argument("--online-semantic-model", default="gpt-4o-mini")
    p.add_argument("--online-semantic-timeout-sec", type=float, default=12.0)
    p.add_argument("--online-semantic-max-checks", type=int, default=12)
    p.add_argument("--online-semantic-on-warn-only", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--online-semantic-base-url", default=None)
    p.add_argument("--online-semantic-api-key", default=None)
    p.add_argument("--sources-config", default="config/sources.yaml")
    p.add_argument("--out", default=None)
    p.set_defaults(func=cmd_query)

    p = sub.add_parser("query-router-eval", help="evaluate deterministic router against a labeled case set")
    p.add_argument("--cases", default="tests/fixtures/router_eval_cases.json")
    p.add_argument("--config", default="config/router.yaml")
    p.add_argument("--out", default="runs/audit/router_eval.json")
    p.add_argument("--strict-min-accuracy", type=float, default=None)
    p.set_defaults(func=cmd_query_router_eval)

    p = sub.add_parser("live-fetch", help="fetch one configured live source and snapshot it")
    p.add_argument("--source", required=True, help="Source name from live_sources config")
    p.add_argument("--query", required=True)
    p.add_argument("--params", action="append", default=[], help="Connector parameters as key=value (repeatable)")
    p.add_argument("--sources-config", default="config/sources.yaml")
    p.add_argument("--out", default=None)
    p.add_argument("--live-max-items", type=int, default=20)
    p.add_argument("--live-timeout-sec", type=int, default=20)
    p.add_argument("--live-cache-ttl-sec", type=int, default=None)
    p.set_defaults(func=cmd_live_fetch)

    p = sub.add_parser("kb-ingest", help="ingest validated report claims into long-lived KB")
    p.add_argument("--report", required=True)
    p.add_argument("--evidence", required=True)
    p.add_argument("--metrics", default=None)
    p.add_argument("--kb-db", default="data/kb/knowledge.db")
    p.add_argument("--topic", required=True)
    p.add_argument("--run-id", default=None)
    p.add_argument("--out", default=None)
    p.set_defaults(func=cmd_kb_ingest)

    p = sub.add_parser("kb-query", help="query long-lived KB claims")
    p.add_argument("--kb-db", default="data/kb/knowledge.db")
    p.add_argument("--query", required=True)
    p.add_argument("--topic", default=None)
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--out", default=None)
    p.set_defaults(func=cmd_kb_query)

    p = sub.add_parser("kb-diff", help="show KB claim changes since run_id/timestamp")
    p.add_argument("--kb-db", default="data/kb/knowledge.db")
    p.add_argument("--topic", required=True)
    p.add_argument("--since-run", default=None)
    p.add_argument("--out", default=None)
    p.set_defaults(func=cmd_kb_diff)

    p = sub.add_parser("kb-backfill", help="ingest last N reports into KB")
    p.add_argument("--kb-db", default="data/kb/knowledge.db")
    p.add_argument("--reports-dir", default="runs/research_reports")
    p.add_argument("--topic", required=True)
    p.add_argument("--last-n", type=int, default=20)
    p.add_argument("--out", default=None)
    p.set_defaults(func=cmd_kb_backfill)

    p = sub.add_parser("kb-contradiction-resolve", help="resolve disputed KB contradiction pairs via targeted follow-up research")
    p.add_argument("--kb-db", default="data/kb/knowledge.db")
    p.add_argument("--topic", required=True)
    p.add_argument("--index", required=True)
    p.add_argument("--out-dir", default="runs/audit/kb_resolution")
    p.add_argument("--out", default=None, help="Optional path to write resolver payload JSON")
    p.add_argument("--run-id", default=None)
    p.add_argument("--max-pairs", type=int, default=5)
    p.add_argument("--support-margin", type=float, default=0.05)
    p.add_argument("--top-k", type=int, default=8)
    p.add_argument("--min-items", type=int, default=2)
    p.add_argument("--min-score", type=float, default=0.0)
    p.add_argument("--retrieval-mode", choices=["lexical", "vector", "hybrid"], default="hybrid")
    p.add_argument("--alpha", type=float, default=0.6)
    p.add_argument("--max-per-paper", type=int, default=2)
    p.add_argument("--quality-prior-weight", type=float, default=0.15)
    p.add_argument("--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2")
    p.add_argument("--sources-config", default="config/sources.yaml")
    p.add_argument("--papers-db", default=None)
    p.add_argument("--reliability-db", default=None)
    p.set_defaults(func=cmd_kb_contradiction_resolve)

    p = sub.add_parser("benchmark-research", help="benchmark research runtime and cache effects")
    p.add_argument("--index", default="data/indexes/bm25_index.json")
    p.add_argument("--query", action="append")
    p.add_argument("--queries-file", default=None)
    p.add_argument("--runs-per-query", type=int, default=3)
    p.add_argument("--top-k", type=int, default=8)
    p.add_argument("--min-items", type=int, default=2)
    p.add_argument("--min-score", type=float, default=0.5)
    p.add_argument("--retrieval-mode", choices=["lexical", "vector", "hybrid"], default="lexical")
    p.add_argument("--alpha", type=float, default=0.6)
    p.add_argument("--max-per-paper", type=int, default=2)
    p.add_argument("--vector-index-path", default=None)
    p.add_argument("--vector-metadata-path", default=None)
    p.add_argument("--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2")
    p.add_argument("--quality-prior-weight", type=float, default=0.15)
    p.add_argument("--vector-service-endpoint", default=None)
    p.add_argument("--vector-nprobe", type=int, default=16)
    p.add_argument("--vector-ef-search", type=int, default=64)
    p.add_argument("--vector-topk-candidate-multiplier", type=float, default=1.5)
    p.add_argument("--semantic-mode", choices=["offline", "online", "hybrid"], default="offline")
    p.add_argument("--semantic-model", default="sentence-transformers/all-MiniLM-L6-v2")
    p.add_argument("--semantic-min-support", type=float, default=0.55)
    p.add_argument("--semantic-max-contradiction", type=float, default=0.30)
    p.add_argument("--semantic-shadow-mode", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--semantic-fail-on-low-support", action="store_true")
    p.add_argument("--online-semantic-model", default="gpt-4o-mini")
    p.add_argument("--online-semantic-timeout-sec", type=float, default=12.0)
    p.add_argument("--online-semantic-max-checks", type=int, default=12)
    p.add_argument("--online-semantic-on-warn-only", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--online-semantic-base-url", default=None)
    p.add_argument("--online-semantic-api-key", default=None)
    p.add_argument(
        "--sources-config",
        default="config/sources.yaml",
        help="Sources YAML config for ranking/limits (default: config/sources.yaml)",
    )
    p.add_argument("--out", default="runs/research_reports/benchmark.json")
    p.set_defaults(func=cmd_benchmark_research)

    p = sub.add_parser("benchmark-scale", help="benchmark retrieval/report on synthetic larger corpus")
    p.add_argument("--corpus", default="data/extracted")
    p.add_argument("--query", action="append")
    p.add_argument("--queries-file", default=None)
    p.add_argument("--repeat-factor", type=int, default=20)
    p.add_argument("--runs-per-query", type=int, default=2)
    p.add_argument("--top-k", type=int, default=8)
    p.add_argument("--out", default="runs/research_reports/benchmark_scale.json")
    p.set_defaults(func=cmd_benchmark_scale)

    p = sub.add_parser("vector-service-build", help="build vector sidecars from lexical index with ANN/sharding options")
    p.add_argument("--index", default="data/indexes/bm25_index.json")
    p.add_argument("--vector-index-path", default=None)
    p.add_argument("--vector-metadata-path", default=None)
    p.add_argument("--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--vector-index-type", choices=["flat", "ivf_flat", "hnsw"], default="flat")
    p.add_argument("--vector-nlist", type=int, default=1024)
    p.add_argument("--vector-m", type=int, default=32)
    p.add_argument("--vector-ef-construction", type=int, default=200)
    p.add_argument("--vector-shards", type=int, default=1)
    p.add_argument("--vector-train-sample-size", type=int, default=200000)
    p.set_defaults(func=cmd_vector_service_build)

    p = sub.add_parser("vector-service-query", help="query vector sidecars directly for debugging")
    p.add_argument("--index", default="data/indexes/bm25_index.json")
    p.add_argument("--question", required=True)
    p.add_argument("--top-k", type=int, default=8)
    p.add_argument("--vector-index-path", default=None)
    p.add_argument("--vector-metadata-path", default=None)
    p.add_argument("--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2")
    p.add_argument("--vector-nprobe", type=int, default=16)
    p.add_argument("--vector-ef-search", type=int, default=64)
    p.set_defaults(func=cmd_vector_service_query)

    p = sub.add_parser("vector-service-health", help="check vector sidecar readiness and metadata")
    p.add_argument("--index", default="data/indexes/bm25_index.json")
    p.add_argument("--vector-index-path", default=None)
    p.add_argument("--vector-metadata-path", default=None)
    p.set_defaults(func=cmd_vector_service_health)

    p = sub.add_parser("vector-service-serve", help="start local vector HTTP service (/health, /query)")
    p.add_argument("--index", default="data/indexes/bm25_index.json")
    p.add_argument("--vector-index-path", default=None)
    p.add_argument("--vector-metadata-path", default=None)
    p.add_argument("--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2")
    p.add_argument("--vector-nprobe", type=int, default=16)
    p.add_argument("--vector-ef-search", type=int, default=64)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    p.set_defaults(func=cmd_vector_service_serve)

    p = sub.add_parser("corpus-health", help="evaluate extracted corpus health before indexing")
    p.add_argument("--corpus", default="data/extracted")
    p.add_argument("--extract-stats", default=None)
    p.add_argument("--min-snippets", type=int, default=1)
    p.add_argument("--min-avg-chars-per-paper", type=int, default=500)
    p.add_argument("--min-avg-chars-per-page", type=int, default=80)
    p.add_argument("--max-extract-error-rate", type=float, default=0.8)
    p.add_argument("--fail-on-unhealthy", action="store_true")
    p.set_defaults(func=cmd_corpus_health)

    p = sub.add_parser("extraction-quality-report", help="generate per-PDF extraction quality report")
    p.add_argument("--corpus", default="data/extracted")
    p.add_argument("--out", default="runs/research_reports/extraction_quality.md")
    p.set_defaults(func=cmd_extraction_quality_report)

    p = sub.add_parser("extraction-eval", help="evaluate extraction fidelity against a gold checks file")
    p.add_argument("--corpus", default="data/extracted")
    p.add_argument("--gold", required=True)
    p.add_argument("--out", default="runs/research_reports/extraction_eval.md")
    p.add_argument("--fail-below", type=float, default=None)
    p.set_defaults(func=cmd_extraction_eval)

    p = sub.add_parser("extraction-eval-scaffold-gold", help="create a starter extraction-gold file from corpus")
    p.add_argument("--corpus", default="data/extracted")
    p.add_argument("--out", required=True)
    p.add_argument("--max-papers", type=int, default=20)
    p.add_argument("--checks-per-paper", type=int, default=2)
    p.add_argument("--min-chars", type=int, default=500)
    p.set_defaults(func=cmd_extraction_eval_scaffold_gold)

    p = sub.add_parser("run-pipeline", help="one-command sync->extract->health->index->research->validate pipeline")
    p.add_argument("--query", required=True)
    p.add_argument("--max-results", type=int, default=20)
    p.add_argument("--db-path", default="data/papers.db")
    p.add_argument("--papers-dir", default="data/papers")
    p.add_argument("--extracted-dir", default="data/extracted")
    p.add_argument("--index", default="data/indexes/bm25_index.json")
    p.add_argument("--report", default="runs/research_reports/pipeline_report.md")
    p.add_argument("--with-semantic-scholar", action="store_true")
    p.add_argument("--top-k", type=int, default=8)
    p.add_argument("--min-items", type=int, default=2)
    p.add_argument("--min-score", type=float, default=0.5)
    p.add_argument("--min-text-chars", type=int, default=200)
    p.add_argument("--require-healthy-corpus", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--min-snippets", type=int, default=1)
    p.add_argument("--min-avg-chars-per-paper", type=int, default=500)
    p.add_argument("--min-avg-chars-per-page", type=int, default=80)
    p.add_argument("--max-extract-error-rate", type=float, default=0.8)
    p.add_argument("--fail-on-source-quality-gate", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--extraction-gold", default=None)
    p.add_argument("--extraction-eval-fail-below", type=float, default=None)
    p.add_argument("--sources-config", default="config/sources.yaml")
    p.add_argument("--summary-out", default="runs/research_reports/pipeline_latest.json")
    p.add_argument("--dispatch-alerts", action="store_true")
    p.add_argument("--alerts-config", default="config/automation.yaml")
    p.add_argument("--alert-out", default="runs/audit/latest_alert_pipeline.json")
    p.set_defaults(func=cmd_run_pipeline)

    p = sub.add_parser("run-automation", help="run scheduled automation workflow defined in automation config")
    p.add_argument("--config", default="config/automation.yaml")
    p.set_defaults(func=cmd_run_automation)

    p = sub.add_parser("serve-ui", help="start FastAPI operator UI")
    p.add_argument("--config", default="config/ui.yaml")
    p.add_argument("--host", default=None)
    p.add_argument("--port", type=int, default=None)
    p.add_argument("--reload", action="store_true")
    p.set_defaults(func=cmd_serve_ui)

    p = sub.add_parser("research-template", help="run a multi-query research template and persist a session pack")
    p.add_argument("--template", required=True)
    p.add_argument("--topic", required=True)
    p.add_argument("--index", required=True)
    p.add_argument("--out-dir", default=None)
    p.add_argument("--templates-config", default="config/templates.yaml")
    p.add_argument("--session-db", default=DEFAULT_SESSION_DB)
    p.add_argument("--sources-config", default="config/sources.yaml")
    p.add_argument("--top-k", type=int, default=8)
    p.add_argument("--min-items", type=int, default=2)
    p.add_argument("--min-score", type=float, default=0.5)
    p.add_argument("--retrieval-mode", choices=["lexical", "vector", "hybrid"], default="hybrid")
    p.add_argument("--alpha", type=float, default=0.6)
    p.add_argument("--max-per-paper", type=int, default=2)
    p.add_argument("--quality-prior-weight", type=float, default=0.15)
    p.add_argument("--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2")
    p.set_defaults(func=cmd_research_template)

    p = sub.add_parser("watch-ingest", help="watch a PDF folder and trigger extract+incremental-index on change")
    p.add_argument("--dir", required=True, help="Directory to watch for new PDFs")
    p.add_argument("--extracted-dir", default="data/extracted")
    p.add_argument("--db-path", default="data/papers.db")
    p.add_argument("--index", default="data/indexes/bm25_index.json")
    p.add_argument("--once", action="store_true", help="Run one scan cycle and exit")
    p.add_argument("--poll-interval-sec", type=float, default=2.0)
    p.add_argument("--max-events", type=int, default=0, help="Stop after processing N changed files (0 = unlimited)")
    p.add_argument("--min-text-chars", type=int, default=200)
    p.add_argument("--out", default=None)
    p.set_defaults(func=cmd_watch_ingest)

    p = sub.add_parser("benchmark-regression-check", help="compare benchmark result against history and emit regression status")
    p.add_argument("--benchmark", required=True, help="Path to benchmark JSON output")
    p.add_argument("--history", default="runs/audit/benchmark_history.json")
    p.add_argument("--run-id", default=None)
    p.add_argument("--max-latency-regression-pct", type=float, default=10.0)
    p.add_argument("--min-quality-floor", type=float, default=0.0)
    p.add_argument("--history-window", type=int, default=104)
    p.add_argument("--fail-on-regression", action="store_true")
    p.add_argument("--out", default=None)
    p.set_defaults(func=cmd_benchmark_regression_check)

    p = sub.add_parser("reliability-watchdog", help="ingest sync stats and update source reliability health")
    p.add_argument("--config", default=None, help="Optional automation config for reliability defaults")
    p.add_argument("--run-dir", default=None, help="Automation run directory containing manifest.json")
    p.add_argument("--sync-stats", action="append", default=[], help="Path(s) to sync stats JSON files")
    p.add_argument("--run-id", default=None)
    p.add_argument("--reliability-db", default=None)
    p.add_argument("--state-path", default=None)
    p.add_argument("--report-path", default=None)
    p.add_argument("--degrade-threshold", type=float, default=None)
    p.add_argument("--critical-threshold", type=float, default=None)
    p.add_argument("--auto-disable-after", type=int, default=None)
    p.add_argument("--out", default=None)
    p.set_defaults(func=cmd_reliability_watchdog)

    p = sub.add_parser("layout-promotion-gate", help="evaluate legacy vs v2 extraction and update layout promotion state")
    p.add_argument("--papers-dir", required=True)
    p.add_argument("--gold", required=True)
    p.add_argument("--state-path", default="runs/audit/layout_promotion_state.json")
    p.add_argument("--db-path", default=None)
    p.add_argument("--min-text-chars", type=int, default=200)
    p.add_argument("--two-column-mode", choices=["off", "auto", "force"], default="auto")
    p.add_argument("--min-weighted-score", type=float, default=0.75)
    p.add_argument("--max-weighted-regression", type=float, default=0.02)
    p.add_argument("--max-ordered-regression", type=float, default=0.02)
    p.add_argument("--max-page-nonempty-regression", type=float, default=0.02)
    p.set_defaults(func=cmd_layout_promotion_gate)

    p = sub.add_parser("migrate-extraction-meta", help="backfill extraction_meta for legacy extracted corpus JSON")
    p.add_argument("--corpus", default="data/extracted")
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=cmd_migrate_extraction_meta)

    p = sub.add_parser("snapshot-run", help="create reproducibility bundle for a report run")
    p.add_argument("--out", default="runs/snapshots")
    p.add_argument("--report", default=None)
    p.add_argument("--index", default=None)
    p.add_argument("--config", action="append")
    p.set_defaults(func=cmd_snapshot_run)

    p = sub.add_parser("doc-write", help="doc-write --facts path --doc-type readme|arch|api")
    p.add_argument("--facts", required=True)
    p.add_argument("--doc-type", choices=["readme", "arch", "api"], required=True)
    p.add_argument("--out", default=None)
    p.set_defaults(func=cmd_doc_write)

    p = sub.add_parser("release-notes", help="release-notes --from <tag> --to <tag>")
    p.add_argument("--from", dest="from_ref", required=True)
    p.add_argument("--to", dest="to_ref", required=True)
    p.add_argument("--out", default="CHANGELOG.generated.md")
    p.set_defaults(func=cmd_release_notes)

    p = sub.add_parser("train-tinygpt", help="train-tinygpt --config config/train.yaml")
    p.add_argument("--data", default="data/data.txt")
    p.add_argument("--out-dir", default="runs")
    p.add_argument("--config", default=None)
    p.set_defaults(func=cmd_train_tinygpt)

    p = sub.add_parser("generate", help="generate --checkpoint ... --prompt ...")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--prompt", required=True)
    p.add_argument("--max-new-tokens", type=int, default=200)
    p.add_argument("--temperature", type=float, default=0.9)
    p.add_argument("--top-k", type=int, default=40)
    p.add_argument("--deterministic", action="store_true")
    p.add_argument("--seed", type=int, default=1337)
    p.set_defaults(func=cmd_generate)

    p = sub.add_parser("validate-report", help="validate-report --input report.md --evidence evidence_pack.json")
    p.add_argument("--input", required=True)
    p.add_argument("--evidence", required=True)
    p.add_argument("--semantic-faithfulness", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--semantic-mode", choices=["offline", "online", "hybrid"], default="offline")
    p.add_argument("--semantic-model", default="sentence-transformers/all-MiniLM-L6-v2")
    p.add_argument("--semantic-min-support", type=float, default=0.55)
    p.add_argument("--semantic-max-contradiction", type=float, default=0.30)
    p.add_argument("--semantic-shadow-mode", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--semantic-fail-on-low-support", action="store_true")
    p.add_argument("--online-semantic-model", default="gpt-4o-mini")
    p.add_argument("--online-semantic-timeout-sec", type=float, default=12.0)
    p.add_argument("--online-semantic-max-checks", type=int, default=12)
    p.add_argument("--online-semantic-on-warn-only", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--online-semantic-base-url", default=None)
    p.add_argument("--online-semantic-api-key", default=None)
    p.set_defaults(func=cmd_validate_report)

    p = sub.add_parser("export-report", help="export a research report to BibTeX / LaTeX / Markdown")
    p.add_argument("--input", required=True, help="Path to report .json file")
    p.add_argument("--evidence", default=None, help="Path to .evidence.json file (optional, enriches output)")
    p.add_argument("--format", default="bibtex", help="Export format(s): bibtex, latex, markdown (comma-separated for multi)")
    p.add_argument("--papers-db", default=None, help="Path to papers SQLite DB for full metadata resolution")
    p.add_argument("--out", default=None, help="Output file path (or directory for multi-format)")
    p.set_defaults(func=cmd_export_report)

    p = sub.add_parser("scaffold-domain-config", help="create a new domain sources config template")
    p.add_argument("--domain", required=True, help='Domain name, e.g., "marketing growth"')
    p.add_argument("--out", default=None, help="Optional explicit output file path")
    p.add_argument("--profile", choices=["auto", "balanced", "academic", "industry"], default="auto")
    p.add_argument("--overwrite", action="store_true")
    p.set_defaults(func=cmd_scaffold_domain_config)

    p = sub.add_parser("monitor-evaluate", help="evaluate monitoring rules for one topic against baseline")
    p.add_argument("--config", default="config/automation.yaml")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--topic", required=True)
    p.add_argument("--out", default=None)
    p.set_defaults(func=cmd_monitor_evaluate)

    p = sub.add_parser("monitor-digest-flush", help="flush queued medium severity monitor digest events")
    p.add_argument("--config", default="config/automation.yaml")
    p.set_defaults(func=cmd_monitor_digest_flush)

    p = sub.add_parser("monitor-status", help="show monitoring state and digest queue status")
    p.add_argument("--config", default="config/automation.yaml")
    p.set_defaults(func=cmd_monitor_status)

    p = sub.add_parser("monitor-soak-sim", help="simulate long-run monitoring behavior (cooldown/hysteresis)")
    p.add_argument("--topic", default="soak_topic")
    p.add_argument("--runs", type=int, default=72)
    p.add_argument("--interval-minutes", type=int, default=60)
    p.add_argument("--cooldown-minutes", type=int, default=360)
    p.add_argument("--hysteresis-runs", type=int, default=2)
    p.add_argument("--pattern", choices=["constant_bad", "pulse", "burst"], default="constant_bad")
    p.add_argument("--trigger-every", type=int, default=4, help="Used by pulse pattern")
    p.add_argument("--burst-len", type=int, default=2, help="Used by burst pattern")
    p.add_argument("--gap-len", type=int, default=4, help="Used by burst pattern")
    p.add_argument("--severity", choices=["low", "medium", "high"], default="high")
    p.add_argument("--metric", default="evidence_usage")
    p.add_argument("--op", choices=["lt", "lte", "gt", "gte", "eq", "neq"], default="lt")
    p.add_argument("--threshold", type=float, default=0.6)
    p.add_argument("--bad-value", type=float, default=0.2)
    p.add_argument("--good-value", type=float, default=0.9)
    p.add_argument("--out", default="runs/audit/monitor_soak_sim.json")
    p.set_defaults(func=cmd_monitor_soak_sim)

    p = sub.add_parser("monitor-history-check", help="analyze real monitoring trigger history from audit runs")
    p.add_argument("--topic", required=True)
    p.add_argument("--audit-dir", default="runs/audit")
    p.add_argument("--out", default="runs/audit/monitor_history_check.json")
    p.set_defaults(func=cmd_monitor_history_check)

    # -- Session tracking subcommands --
    p = sub.add_parser("session-create", help="create a new research session")
    p.add_argument("--name", required=True, help="Unique session name")
    p.add_argument("--description", default="", help="Optional session description")
    p.add_argument("--session-db", default=DEFAULT_SESSION_DB)
    p.set_defaults(func=cmd_session_create)

    p = sub.add_parser("session-list", help="list research sessions")
    p.add_argument("--status", choices=["open", "closed", "all"], default="all")
    p.add_argument("--session-db", default=DEFAULT_SESSION_DB)
    p.set_defaults(func=cmd_session_list)

    p = sub.add_parser("session-show", help="show session details")
    p.add_argument("--session-id", type=int, required=True)
    p.add_argument("--format", choices=["markdown", "json"], default="markdown")
    p.add_argument("--session-db", default=DEFAULT_SESSION_DB)
    p.set_defaults(func=cmd_session_show)

    p = sub.add_parser("session-summary", help="generate cross-query session summary")
    p.add_argument("--session-id", type=int, required=True)
    p.add_argument("--format", choices=["markdown", "json"], default="markdown")
    p.add_argument("--out", default=None, help="Optional path to write summary Markdown")
    p.add_argument("--session-db", default=DEFAULT_SESSION_DB)
    p.set_defaults(func=cmd_session_summary)

    p = sub.add_parser("session-close", help="close a research session")
    p.add_argument("--session-id", type=int, required=True)
    p.add_argument("--session-db", default=DEFAULT_SESSION_DB)
    p.set_defaults(func=cmd_session_close)

    p = sub.add_parser("session-reopen", help="reopen a closed research session")
    p.add_argument("--session-id", type=int, required=True)
    p.add_argument("--session-db", default=DEFAULT_SESSION_DB)
    p.set_defaults(func=cmd_session_reopen)

    p = sub.add_parser("session-delete", help="delete a session and all its queries")
    p.add_argument("--session-id", type=int, required=True)
    p.add_argument("--session-db", default=DEFAULT_SESSION_DB)
    p.set_defaults(func=cmd_session_delete)

    # --- source reliability commands ---
    p = sub.add_parser("reliability-list", help="list all sources with reliability scores")
    p.add_argument("--sort-by", default="reliability_score",
                   choices=["reliability_score", "source_name", "total_events", "created_at"])
    p.add_argument("--format", choices=["table", "json"], default="table")
    p.add_argument("--reliability-db", default=DEFAULT_RELIABILITY_DB)
    p.set_defaults(func=cmd_reliability_list)

    p = sub.add_parser("reliability-show", help="show detailed reliability for one source")
    p.add_argument("--source", required=True, help="Source connector name (e.g. arxiv)")
    p.add_argument("--events", type=int, default=20, help="Number of recent events to show")
    p.add_argument("--format", choices=["text", "json"], default="text")
    p.add_argument("--reliability-db", default=DEFAULT_RELIABILITY_DB)
    p.set_defaults(func=cmd_reliability_show)

    p = sub.add_parser("reliability-record", help="record a feedback event for a source")
    p.add_argument("--source", required=True, help="Source connector name")
    p.add_argument("--event", required=True, choices=sorted(EVENT_TYPES), help="Event type")
    p.add_argument("--value", type=float, default=1.0, help="Event value (default 1.0)")
    p.add_argument("--run-id", default=None, help="Optional run identifier")
    p.add_argument("--reliability-db", default=DEFAULT_RELIABILITY_DB)
    p.set_defaults(func=cmd_reliability_record)

    p = sub.add_parser("reliability-recompute", help="recompute reliability scores from events")
    p.add_argument("--source", default=None, help="Recompute one source (default: all)")
    p.add_argument("--reliability-db", default=DEFAULT_RELIABILITY_DB)
    p.set_defaults(func=cmd_reliability_recompute)

    p = sub.add_parser("reliability-report", help="generate full source reliability report")
    p.add_argument("--format", choices=["markdown", "json"], default="markdown")
    p.add_argument("--out", default=None, help="Optional output file path")
    p.add_argument("--reliability-db", default=DEFAULT_RELIABILITY_DB)
    p.set_defaults(func=cmd_reliability_report)

    p = sub.add_parser("reliability-delete", help="delete a source and all its events")
    p.add_argument("--source", required=True, help="Source connector name to delete")
    p.add_argument("--reliability-db", default=DEFAULT_RELIABILITY_DB)
    p.set_defaults(func=cmd_reliability_delete)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
