from src.cli import build_parser


def test_research_parser_default_retrieval_mode() -> None:
    parser = build_parser()
    args = parser.parse_args(
        ["research", "--question", "q", "--index", "data/indexes/x.json", "--out", "runs/research_reports/x.md"]
    )
    assert args.retrieval_mode == "lexical"
    assert args.alpha == 0.6
    assert args.max_per_paper == 2
    assert args.quality_prior_weight == 0.15


def test_sync_parser_sources_config_flag() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "sync-papers",
            "--query",
            "mobile segmentation",
            "--sources-config",
            "config/sources.yaml",
        ]
    )
    assert args.sources_config == "config/sources.yaml"


def test_sync_parser_sources_config_default() -> None:
    parser = build_parser()
    args = parser.parse_args(["sync-papers", "--query", "mobile segmentation"])
    assert args.sources_config == "config/sources.yaml"
    assert args.fail_on_source_quality_gate is False


def test_sync_parser_source_quality_gate_flag() -> None:
    parser = build_parser()
    args = parser.parse_args(["sync-papers", "--query", "mobile segmentation", "--fail-on-source-quality-gate"])
    assert args.fail_on_source_quality_gate is True


def test_research_parser_sources_config_default() -> None:
    parser = build_parser()
    args = parser.parse_args(
        ["research", "--question", "q", "--index", "data/indexes/x.json", "--out", "runs/research_reports/x.md"]
    )
    assert args.sources_config == "config/sources.yaml"


def test_build_index_parser_vector_flags() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "build-index",
            "--corpus",
            "data/extracted",
            "--out",
            "data/indexes/x.json",
            "--with-vector",
            "--embedding-model",
            "sentence-transformers/all-MiniLM-L6-v2",
            "--batch-size",
            "32",
        ]
    )
    assert args.with_vector is True
    assert args.embedding_model == "sentence-transformers/all-MiniLM-L6-v2"
    assert args.batch_size == 32
    assert args.vector_index_type == "flat"
    assert args.vector_nlist == 1024
    assert args.vector_m == 32
    assert args.vector_ef_construction == 200
    assert args.vector_shards == 1
    assert args.vector_train_sample_size == 200000
    assert args.require_healthy_corpus is False


def test_research_parser_vector_query_flags() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "research",
            "--question",
            "q",
            "--index",
            "data/indexes/x.json",
            "--vector-nprobe",
            "24",
            "--vector-ef-search",
            "96",
            "--vector-topk-candidate-multiplier",
            "2.0",
            "--out",
            "runs/research_reports/x.md",
        ]
    )
    assert args.vector_nprobe == 24
    assert args.vector_ef_search == 96
    assert args.vector_topk_candidate_multiplier == 2.0


def test_extract_parser_new_flags() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "extract-corpus",
            "--papers-dir",
            "data/papers",
            "--out-dir",
            "data/extracted",
            "--two-column-mode",
            "force",
            "--layout-engine",
            "v2",
            "--layout-table-handling",
            "preserve",
            "--layout-footnote-handling",
            "append",
            "--layout-min-region-confidence",
            "0.7",
            "--ocr-enabled",
            "--ocr-timeout-sec",
            "10",
            "--ocr-min-chars-trigger",
            "80",
            "--ocr-max-pages",
            "5",
            "--ocr-min-output-chars",
            "250",
            "--ocr-min-gain-chars",
            "70",
            "--ocr-min-confidence",
            "55.5",
            "--ocr-lang",
            "eng+deu",
            "--ocr-profile",
            "sparse",
            "--no-ocr-noise-suppression",
        ]
    )
    assert args.two_column_mode == "force"
    assert args.layout_engine == "v2"
    assert args.layout_table_handling == "preserve"
    assert args.layout_footnote_handling == "append"
    assert args.layout_min_region_confidence == 0.7
    assert args.ocr_enabled is True
    assert args.ocr_timeout_sec == 10
    assert args.ocr_min_chars_trigger == 80
    assert args.ocr_max_pages == 5
    assert args.ocr_min_output_chars == 250
    assert args.ocr_min_gain_chars == 70
    assert args.ocr_min_confidence == 55.5
    assert args.ocr_lang == "eng+deu"
    assert args.ocr_profile == "sparse"
    assert args.ocr_noise_suppression is False
    assert args.sources_config == "config/sources.yaml"
    assert args.layout_promotion_state == "runs/audit/layout_promotion_state.json"


def test_extract_parser_boolean_optional_ocr_enabled() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "extract-corpus",
            "--papers-dir",
            "data/papers",
            "--out-dir",
            "data/extracted",
            "--no-ocr-enabled",
        ]
    )
    assert args.ocr_enabled is False


def test_extract_parser_layout_engine_auto() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "extract-corpus",
            "--papers-dir",
            "data/papers",
            "--out-dir",
            "data/extracted",
            "--layout-engine",
            "auto",
            "--layout-promotion-state",
            "runs/audit/custom_layout_state.json",
        ]
    )
    assert args.layout_engine == "auto"
    assert args.layout_promotion_state.endswith("custom_layout_state.json")


def test_layout_promotion_gate_parser_flags() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "layout-promotion-gate",
            "--papers-dir",
            "data/papers",
            "--gold",
            "tests/fixtures/extraction_gold.json",
            "--state-path",
            "runs/audit/layout_promotion_state.json",
            "--min-weighted-score",
            "0.8",
        ]
    )
    assert args.papers_dir == "data/papers"
    assert args.gold.endswith("extraction_gold.json")
    assert args.state_path.endswith("layout_promotion_state.json")
    assert args.min_weighted_score == 0.8


def test_corpus_health_parser_flags() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "corpus-health",
            "--corpus",
            "data/extracted",
            "--extract-stats",
            "runs/audit/extract.stats.json",
            "--min-snippets",
            "3",
            "--min-avg-chars-per-paper",
            "600",
            "--min-avg-chars-per-page",
            "90",
            "--max-extract-error-rate",
            "0.4",
            "--fail-on-unhealthy",
        ]
    )
    assert args.corpus == "data/extracted"
    assert args.extract_stats.endswith("extract.stats.json")
    assert args.min_snippets == 3
    assert args.min_avg_chars_per_paper == 600
    assert args.min_avg_chars_per_page == 90
    assert args.max_extract_error_rate == 0.4
    assert args.fail_on_unhealthy is True


def test_build_index_parser_health_flags() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "build-index",
            "--corpus",
            "data/extracted",
            "--out",
            "data/indexes/x.json",
            "--require-healthy-corpus",
            "--min-snippets",
            "2",
            "--min-avg-chars-per-paper",
            "450",
            "--min-avg-chars-per-page",
            "90",
            "--max-extract-error-rate",
            "0.3",
        ]
    )
    assert args.require_healthy_corpus is True
    assert args.min_snippets == 2
    assert args.min_avg_chars_per_paper == 450
    assert args.min_avg_chars_per_page == 90
    assert args.max_extract_error_rate == 0.3


def test_extraction_quality_report_parser_flags() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "extraction-quality-report",
            "--corpus",
            "data/extracted",
            "--out",
            "runs/research_reports/extraction_quality.md",
        ]
    )
    assert args.corpus == "data/extracted"
    assert args.out.endswith("extraction_quality.md")


def test_extraction_eval_parser_flags() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "extraction-eval",
            "--corpus",
            "data/extracted",
            "--gold",
            "tests/fixtures/extraction_gold.json",
            "--out",
            "runs/research_reports/extraction_eval.md",
            "--fail-below",
            "0.8",
        ]
    )
    assert args.corpus == "data/extracted"
    assert args.gold.endswith("extraction_gold.json")
    assert args.out.endswith("extraction_eval.md")
    assert args.fail_below == 0.8


def test_extraction_eval_scaffold_parser_flags() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "extraction-eval-scaffold-gold",
            "--corpus",
            "data/extracted",
            "--out",
            "runs/research_reports/extraction_gold.generated.json",
            "--max-papers",
            "12",
            "--checks-per-paper",
            "3",
            "--min-chars",
            "600",
        ]
    )
    assert args.corpus == "data/extracted"
    assert args.out.endswith("extraction_gold.generated.json")
    assert args.max_papers == 12
    assert args.checks_per_paper == 3
    assert args.min_chars == 600


def test_run_pipeline_parser_flags() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "run-pipeline",
            "--query",
            "mobile segmentation",
            "--fail-on-source-quality-gate",
            "--no-require-healthy-corpus",
            "--extraction-gold",
            "tests/fixtures/extraction_gold.json",
            "--extraction-eval-fail-below",
            "0.7",
        ]
    )
    assert args.query == "mobile segmentation"
    assert args.fail_on_source_quality_gate is True
    assert args.require_healthy_corpus is False
    assert args.extraction_gold.endswith("extraction_gold.json")
    assert args.extraction_eval_fail_below == 0.7
    assert args.summary_out.endswith("pipeline_latest.json")
    assert args.dispatch_alerts is False
    assert args.alerts_config.endswith("automation.yaml")
    assert args.alert_out.endswith("latest_alert_pipeline.json")


def test_run_pipeline_parser_alert_flags() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "run-pipeline",
            "--query",
            "mobile segmentation",
            "--dispatch-alerts",
            "--alerts-config",
            "config/automation.yaml",
            "--alert-out",
            "runs/audit/custom_alert.json",
        ]
    )
    assert args.dispatch_alerts is True
    assert args.alerts_config == "config/automation.yaml"
    assert args.alert_out.endswith("custom_alert.json")


def test_migrate_extraction_meta_parser_flags() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "migrate-extraction-meta",
            "--corpus",
            "data/extracted",
            "--dry-run",
        ]
    )
    assert args.corpus == "data/extracted"
    assert args.dry_run is True


def test_scaffold_domain_config_parser_flags() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "scaffold-domain-config",
            "--domain",
            "Marketing Growth",
            "--profile",
            "industry",
            "--overwrite",
        ]
    )
    assert args.domain == "Marketing Growth"
    assert args.profile == "industry"
    assert args.overwrite is True


def test_research_parser_live_flags() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "research",
            "--question",
            "q",
            "--index",
            "data/indexes/x.json",
            "--live",
            "--live-sources",
            "a,b",
            "--live-max-items",
            "11",
            "--live-timeout-sec",
            "9",
            "--live-cache-ttl-sec",
            "15",
            "--live-merge-mode",
            "live_first",
            "--out",
            "runs/research_reports/x.md",
        ]
    )
    assert args.live is True
    assert args.live_sources == "a,b"
    assert args.live_max_items == 11
    assert args.live_timeout_sec == 9
    assert args.live_cache_ttl_sec == 15
    assert args.live_merge_mode == "live_first"


def test_live_fetch_parser_flags() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "live-fetch",
            "--source",
            "demo",
            "--query",
            "q",
            "--params",
            "symbol=NVDA",
            "--sources-config",
            "config/sources.yaml",
        ]
    )
    assert args.source == "demo"
    assert args.query == "q"
    assert args.params == ["symbol=NVDA"]
    assert args.sources_config == "config/sources.yaml"


def test_build_index_parser_live_data_repeatable() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "build-index",
            "--corpus",
            "data/extracted",
            "--out",
            "data/indexes/x.json",
            "--live-data",
            "runs/live/a.json",
            "--live-data",
            "runs/live/b.json",
        ]
    )
    assert args.live_data == ["runs/live/a.json", "runs/live/b.json"]


def test_research_parser_routing_and_direct_answer_flags() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "research",
            "--question",
            "q",
            "--index",
            "data/indexes/x.json",
            "--routing-mode",
            "manual",
            "--intent",
            "weather_now",
            "--relevance-policy",
            "warn",
            "--no-diagnostics",
            "--direct-answer-mode",
            "off",
            "--direct-answer-max-complexity",
            "5",
            "--out",
            "runs/research_reports/x.md",
        ]
    )
    assert args.routing_mode == "manual"
    assert args.intent == "weather_now"
    assert args.relevance_policy == "warn"
    assert args.diagnostics is False
    assert args.direct_answer_mode == "off"
    assert args.direct_answer_max_complexity == 5


def test_research_parser_kb_flags() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "research",
            "--question",
            "q",
            "--index",
            "data/indexes/x.json",
            "--use-kb",
            "--kb-db",
            "data/kb/knowledge.db",
            "--kb-top-k",
            "7",
            "--kb-merge-weight",
            "0.2",
            "--out",
            "runs/research_reports/x.md",
        ]
    )
    assert args.use_kb is True
    assert args.kb_db.endswith("knowledge.db")
    assert args.kb_top_k == 7
    assert args.kb_merge_weight == 0.2


def test_kb_parsers_flags() -> None:
    parser = build_parser()
    ingest = parser.parse_args(
        [
            "kb-ingest",
            "--report",
            "runs/research_reports/x.md",
            "--evidence",
            "runs/research_reports/x.evidence.json",
            "--metrics",
            "runs/research_reports/x.metrics.json",
            "--kb-db",
            "data/kb/knowledge.db",
            "--topic",
            "finance_markets",
        ]
    )
    assert ingest.topic == "finance_markets"
    q = parser.parse_args(["kb-query", "--kb-db", "data/kb/knowledge.db", "--query", "drift", "--top-k", "5"])
    assert q.query == "drift"
    d = parser.parse_args(["kb-diff", "--kb-db", "data/kb/knowledge.db", "--topic", "finance_markets"])
    assert d.topic == "finance_markets"
    b = parser.parse_args(["kb-backfill", "--kb-db", "data/kb/knowledge.db", "--topic", "finance_markets", "--last-n", "20"])
    assert b.last_n == 20
