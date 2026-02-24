# Tiny GPT From Scratch (Offline, CPU-First)

This repository documents and implements a tiny GPT-style language model built from scratch using Python and PyTorch.

## Goal
Build a fully offline character-level language model that can:
1. Train on a small text dataset.
2. Save checkpoints.
3. Load checkpoints.
4. Generate text from a prompt.
5. Run on a consumer PC (CPU-compatible).

## Practical Outcomes For Us
We are building the core model to power three narrow, useful offline workflows:
1. Documentation writer:
- Turn structured project facts into README, architecture notes, and API sections in a consistent style.
2. Release notes generator:
- Turn categorized commit/change inputs into polished changelog and release-note text.
3. Research copilot (does research for us):
- Discover papers, rank relevance, extract evidence, synthesize findings, and propose experiments with citations.

Important: the tiny model is a local wording engine, not a world-knowledge engine. For factual tasks, we will feed source text and enforce grounding rules.

## Constraints
- Core transformer implementation is manual (no Hugging Face model classes).
- Learning-first process: small, testable phases.
- Every phase must have verification before proceeding.

## Documentation Index
- `docs/roadmap.md`: Full engineering roadmap and architecture.
- `docs/phases.md`: Phase-by-phase execution plan with milestones.
- `docs/checklist.md`: Exit criteria and final validation checklist.

## Planned Project Structure
```
tinygpt/
  README.md
  requirements.txt
  data/
    data.txt
  src/
    config.py
    tokenizer.py
    dataset.py
    model/
      layers.py
      gpt.py
    train.py
    generate.py
    checkpoint.py
    utils.py
    apps/
      doc_writer.py
      release_notes.py
      research_copilot.py
      retrieval.py
      paper_ingest.py
      paper_search.py
      evidence_extract.py
  tests/
    test_tokenizer.py
    test_dataset.py
    test_shapes.py
    test_save_load.py
    test_apps.py
  runs/
    checkpoints/
    logs/
  docs/
    roadmap.md
    phases.md
    checklist.md
```

## How We Will Work
- We implement in strict order: setup -> data -> model components -> training -> generation -> persistence -> debugging/eval.
- We do not advance phases until verification for the current phase passes.
- After the core model passes validation, we build the three application workflows with strict grounding/validation checks.

## Status
MVP implementation is in place with runnable CLI, tiny GPT core, research retrieval/report pipeline, and tests.

## Tech Stack
- Language/runtime: Python (venv-based local runtime)
- Core ML: PyTorch (tiny GPT training/inference)
- Embeddings: sentence-transformers
- Retrieval:
  - Lexical BM25 (custom in-repo implementation)
  - Vector FAISS (`flat`, `ivf_flat`, `hnsw`, optional sharding)
- Paper/data connectors: requests + source adapters (arXiv, OpenAlex, Semantic Scholar, etc.)
- PDF extraction: pypdf (primary), optional pymupdf and pdfminer.six fallbacks
- OCR (optional): Tesseract CLI
- Storage:
  - SQLite (paper metadata)
  - JSON artifacts (extracted corpus, indexes, reports, metrics)
- Interfaces:
  - argparse CLI (`python -m src.cli ...`)
  - optional local vector HTTP service (`/health`, `/query`)
- Config and ops: YAML configs, Makefile workflows, automation runner
- Testing: pytest

## Quickstart
1. Create and activate virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
If extraction hits AES-encrypted PDFs, ensure `cryptography` is installed in the active venv:
```bash
pip install -r requirements.txt
```
2. Verify CLI contract:
```bash
python -m src.cli --help
```
3. Run tests:
```bash
pytest -q
```

## Production Packaging Profiles
Pinned runtime profiles are provided under `deploy/`:
- `deploy/requirements.base.txt`
- `deploy/requirements.cpu.txt`
- `deploy/requirements.cuda12.txt`

CPU-only install:
```bash
pip install -r deploy/requirements.cpu.txt
```

CUDA 12.1 install:
```bash
pip install --extra-index-url https://download.pytorch.org/whl/cu121 -r deploy/requirements.cuda12.txt
```

Container builds:
```bash
make docker-build-cpu
make docker-build-cuda
```
or directly:
```bash
docker build -f docker/Dockerfile.cpu -t yipoodle:cpu .
docker build -f docker/Dockerfile.cuda -t yipoodle:cuda .
```

## Core Command Flow
1. Train tiny GPT:
```bash
python -m src.cli train-tinygpt --data data/data.txt --config config/train.yaml
```
2. Generate from checkpoint:
```bash
python -m src.cli generate --checkpoint runs/checkpoints/<id>.pt --prompt "Research:"
```
Checkpoint loading is hardened to `torch.load(..., weights_only=True)` with strict payload schema validation (`model_state`, `cfg`, `chars`) before model restore.
3. Sync papers (online step):
```bash
python -m src.cli sync-papers --query "mobile segmentation computer vision" --max-results 20 --with-semantic-scholar
```
`config/sources.yaml` is applied by default. To override path explicitly:
```bash
python -m src.cli sync-papers --query "mobile segmentation computer vision" --max-results 20 --with-semantic-scholar --sources-config config/sources.yaml
```
Prebuilt domain configs are available under `config/domains/`:
- `config/domains/sources_nlp.yaml`
- `config/domains/sources_biomed_ai.yaml`
- `config/domains/sources_robotics.yaml`
- `config/domains/sources_cybersecurity.yaml`
- `config/domains/sources_multimodal_ai.yaml`
- `config/domains/sources_speech_audio.yaml`
- `config/domains/sources_reinforcement_learning.yaml`
- `config/domains/sources_data_engineering.yaml`
- `config/domains/sources_cloud_infrastructure.yaml`
- `config/domains/sources_software_engineering.yaml`
- `config/domains/sources_finance_markets.yaml`

Example domain switch:
```bash
python -m src.cli sync-papers --query "retrieval augmented generation" --max-results 20 --with-semantic-scholar --sources-config config/domains/sources_nlp.yaml
```
Applied today from `sources.yaml`:
- `sources.<name>.enabled` (`arxiv`, `openalex`, `semanticscholar`, `crossref`, `dblp`, `paperswithcode`, `core`, `openreview`, `github`, `zenodo`, `opencitations`, `springer`, `ieee_xplore`, `figshare`, `openml`, `gdelt`, `wikidata`, `orcid`)
- `sources.<name>.max_results`
- `limits.max_total_results`
- `limits.max_pdf_downloads`
- `ranking.weights` (mapped into retrieval metadata prior strength)
- `limits.max_tokens_per_summary` (caps report synthesis/gaps/experiment text)
- optional DOI -> OA PDF enrichment via `sources.unpaywall` (`required_query_params.email`)
- optional source-quality gates via `limits.source_quality`:
  - `max_source_error_rate`
  - `min_download_success_rate`
  - `max_download_http_error_rate`
  - `max_non_pdf_content_rate`
  - `max_blocked_or_paywalled_rate`
  - `max_unpaywall_lookup_error_rate`

If unsupported sources are enabled (source key not implemented in code), sync reports:
- `unsupported_enabled_sources_count`
- `unsupported_enabled_sources`

To enable Unpaywall enrichment:
```bash
export UNPAYWALL_EMAIL="you@example.com"
```
`sync-papers` now normalizes DOI forms (`doi:...`, `https://doi.org/...`) before lookup and uses a local cache (`runs/cache/unpaywall.json` by default, configurable via `cache.unpaywall_cache_path`).
4. Extract local PDF corpus:
```bash
python -m src.cli extract-corpus --papers-dir data/papers --out-dir data/extracted --db-path data/papers.db
```
You can apply domain OCR defaults from a sources config:
```bash
python -m src.cli extract-corpus --papers-dir data/papers --out-dir data/extracted --sources-config config/domains/sources_finance_markets.yaml
```
Extraction quality gate:
- `extract-corpus` supports `--min-text-chars` (default `200`) and reports
  `empty_text_skipped` / `low_text_skipped` counters.
- Extraction uses optional fallback chain: `pymupdf` -> `pdfminer.six` -> `pypdf`.
- Two-column reconstruction can be controlled with `--two-column-mode off|auto|force` (default `auto`).
- Layout engine controls:
  - `--layout-engine legacy|v2|shadow|auto` (default `shadow`)
  - `--layout-promotion-state` (default `runs/audit/layout_promotion_state.json`; used by `layout-engine=auto`)
  - `--layout-table-handling drop|linearize|preserve` (default `linearize`)
  - `--layout-footnote-handling drop|append|preserve` (default `append`)
  - `--layout-min-region-confidence` (default `0.55`)
- In `shadow` mode, extraction computes v2 diagnostics but keeps legacy output text.
- Two-column extraction now includes extra layout cleanup for mixed academic layouts:
  - running header/footer suppression
  - figure/table caption suppression
  - y-band reading-order stabilization for column text
- Optional OCR fallback is available for low-text PDFs with:
  - `--ocr-enabled`
  - `--ocr-min-chars-trigger` (default `120`)
  - `--ocr-timeout-sec` (default `30`)
  - `--ocr-max-pages` (default `20`)
  - `--ocr-min-output-chars` (default `200`)
  - `--ocr-min-gain-chars` (default `40`)
  - `--ocr-min-confidence` (default `45.0`)
  - `--ocr-lang` (default `eng`; supports `auto` language detection)
  - `--ocr-profile` (`document|sparse`, default `document`)
  - `--ocr-noise-suppression/--no-ocr-noise-suppression` (default enabled)
- OCR defaults can also be defined in `sources.yaml` (or domain files) under top-level `ocr:`.
  Precedence is: `CLI flags` > `sources-config ocr block` > `built-in defaults`.
- Missing optional extractors do not break execution; pypdf path remains supported.
- New extraction stats include:
  - `failed_pdfs_count`
  - `failed_reason_counts` (e.g., `encrypted_or_unsupported`, `all_extractors_failed`, `empty_text`, `low_text`)
  - `extractor_used_counts`
  - `failed_pdfs` (capped sample list with `path` + `reason`)
  - `two_column_applied_count`
  - `layout_v2_attempted_count`, `layout_v2_applied_count`
  - `layout_shadow_compared_count`, `layout_shadow_diff_rate`
  - `layout_region_counts`
  - `layout_confidence_avg`
  - `layout_fallback_to_legacy_count`
  - `ocr_attempted_count`, `ocr_succeeded_count`, `ocr_failed_count`
  - `ocr_rejected_low_quality_count`
  - `ocr_rejected_low_confidence_count`
  - `ocr_lang_auto_detected_count`
  - `ocr_lang_detected_counts`
  - `ocr_avg_confidence`
  - `quality_band_counts`, `quality_score_avg`, `quality_score_min`, `quality_score_max`

Layout v2 promotion gate (gold-eval + non-regression) and controlled switch:
```bash
python -m src.cli layout-promotion-gate \
  --papers-dir data/papers \
  --gold tests/fixtures/extraction_gold.json \
  --state-path runs/audit/layout_promotion_state.json
```
Then:
```bash
python -m src.cli extract-corpus \
  --papers-dir data/papers \
  --out-dir data/extracted \
  --layout-engine auto \
  --layout-promotion-state runs/audit/layout_promotion_state.json
```
`layout-engine=auto` selects `v2` only when promotion state says it is promoted; otherwise it stays on `shadow`.

Optional extraction extras:
```bash
pip install pymupdf pdfminer.six
```
OCR binary requirement (optional path):
```bash
sudo apt-get install tesseract-ocr
```
5. Build retrieval index:
```bash
python -m src.cli build-index --corpus data/extracted --out data/indexes/bm25_index.json --db-path data/papers.db
```
Enforce corpus-health gate directly inside `build-index` (recommended):
```bash
python -m src.cli build-index --corpus data/extracted --out data/indexes/bm25_index.json --db-path data/papers.db --require-healthy-corpus --min-snippets 1 --min-avg-chars-per-paper 500 --min-avg-chars-per-page 80 --max-extract-error-rate 0.8
```
Corpus health precheck (recommended before index build):
```bash
python -m src.cli corpus-health --corpus data/extracted --extract-stats runs/audit/extract.stats.json --min-snippets 1 --min-avg-chars-per-paper 500 --max-extract-error-rate 0.8
```
You can also enforce page-density quality:
```bash
python -m src.cli corpus-health --corpus data/extracted --min-avg-chars-per-page 80
```
`corpus-health` also reports:
- `papers_with_page_stats`
- `page_stats_coverage_pct`
- `warnings` (e.g., `page_stats_missing_for_corpus`)

Extraction quality report (one command):
```bash
python -m src.cli extraction-quality-report --corpus data/extracted --out runs/research_reports/extraction_quality.md
```
This writes:
- markdown summary (`.md`) with per-PDF stats and spot-check samples
- machine-readable payload (`.json`) with worst pages and OCR/empty-page percentages

Gold-check extraction evaluation harness:
```bash
python -m src.cli extraction-eval --corpus data/extracted --gold tests/fixtures/extraction_gold.json --out runs/research_reports/extraction_eval.md --fail-below 0.75
```
Supported gold check types:
- `contains`
- `ordered_contains` (reading-order proxy)
- `min_chars`
- `page_nonempty_ratio` (per-page extraction fidelity proxy)
The command writes `.md` + `.json` and can fail CI when `weighted_score` is below threshold.

Backfill legacy extracted corpora with missing extraction metadata:
```bash
python -m src.cli migrate-extraction-meta --corpus data/extracted --dry-run
python -m src.cli migrate-extraction-meta --corpus data/extracted
```
This adds `extraction_meta` to older JSON files so quality reporting is consistent.

Using `--db-path` enriches snippet metadata (year/venue/citation_count) from the sync database.
Optional hybrid/vector artifacts:
```bash
python -m src.cli build-index --corpus data/extracted --out data/indexes/bm25_index.json --db-path data/papers.db --with-vector --embedding-model sentence-transformers/all-MiniLM-L6-v2
```
Scalable FAISS build options (ANN + sharding, still offline/local):
```bash
python -m src.cli build-index \
  --corpus data/extracted \
  --out data/indexes/bm25_index.json \
  --with-vector \
  --vector-index-type ivf_flat \
  --vector-nlist 1024 \
  --vector-shards 4 \
  --vector-train-sample-size 200000
```
This writes sidecars next to lexical index:
- `data/indexes/bm25_index.faiss`
- `data/indexes/bm25_index.vector_meta.json`
Live fetch + snapshot (opt-in, allowlisted by `live_sources` in sources config):
```bash
python -m src.cli live-fetch \
  --source yahoo_finance \
  --query "NVDA intraday ticks" \
  --params symbol=NVDA \
  --sources-config config/sources.yaml
```
This writes:
- normalized live snippets payload (`runs/live/<source>/<timestamp>.json`)
- local snapshot cache (`data/live_snapshots/...`)

You can append persisted live snippets into a lexical/vector build:
```bash
python -m src.cli build-index \
  --corpus data/extracted \
  --out data/indexes/bm25_plus_live.json \
  --live-data runs/live/yahoo_finance/<timestamp>.json
```
6. Generate research report (offline from local corpus):
```bash
python -m src.cli research --index data/indexes/bm25_index.json --question "What are current limitations?" --top-k 8 --min-items 2 --min-score 0.5 --out runs/research_reports/report.md
```
Use `--no-cache` to force recomputation, or `--cache-dir` to customize query cache location.
Retrieval mode options:
- `--retrieval-mode lexical` (default)
- `--retrieval-mode vector`
- `--retrieval-mode hybrid --alpha 0.6`
You can also tune diversity with `--max-per-paper`.
Extraction-quality-aware scoring can be tuned with `--quality-prior-weight` (default `0.15`).
This prior is applied in lexical, vector, and hybrid retrieval paths.
ANN query knobs:
- `--vector-nprobe` (IVF query breadth)
- `--vector-ef-search` (HNSW search depth)
- `--vector-topk-candidate-multiplier` (larger vector candidate pool before fusion)
- `--vector-service-endpoint` (optional local/remote `/query` endpoint; falls back to in-process on failure)
Live query knobs (default static-only):
- `--live` (enable live connectors for this query)
- `--live-sources source_a,source_b` (optional override list)
- `--live-max-items` (default `20`)
- `--live-timeout-sec` (default `20`)
- `--live-cache-ttl-sec` (override per-source cache TTL)
- `--live-merge-mode union|live_first` (default `union`)
- `--routing-mode auto|manual` (default `auto`)
- `--intent <name>` (optional forced intent in auto mode)
- `--relevance-policy not_found|warn|fail` (default `not_found`)
- `--diagnostics/--no-diagnostics` (default enabled)
- `--direct-answer-mode off|hybrid` (default `hybrid`)
- `--direct-answer-max-complexity` (default `2`)

`--live` keeps validation/citation guarantees by requiring local snapshot IDs for live evidence snippets (`SNAP:<snapshot_id>:S<idx>`).
`research` now applies:
- intent-routed source packs (from `live_routing` in sources config),
- a question-evidence relevance gate,
- deterministic direct arithmetic answers in hybrid direct-answer mode.

If evidence is off-topic or missing for the detected intent, output is `Not found in sources.` with a `## Retrieval Diagnostics` section.

Long-lived KB (opt-in, grounded memory lane):
- `--use-kb` (advisory merge of KB candidates into retrieval)
- `--kb-db data/kb/knowledge.db`
- `--kb-top-k 5`
- `--kb-merge-weight 0.15`

KB commands:
```bash
python -m src.cli kb-ingest \
  --report runs/research_reports/report.md \
  --evidence runs/research_reports/report.evidence.json \
  --metrics runs/research_reports/report.metrics.json \
  --kb-db data/kb/knowledge.db \
  --topic finance_markets

python -m src.cli kb-query --kb-db data/kb/knowledge.db --query "transformer drift risk" --topic finance_markets --top-k 10
python -m src.cli kb-diff --kb-db data/kb/knowledge.db --topic finance_markets --since-run 20260224T120000Z
python -m src.cli kb-backfill --kb-db data/kb/knowledge.db --reports-dir runs/research_reports --topic finance_markets --last-n 20
```

`kb-ingest` requires a `## Key Claims` section in report markdown. Current report rendering emits that section by default.

7. Validate report grounding:
```bash
python -m src.cli validate-report --input runs/research_reports/report.md --evidence runs/research_reports/report.evidence.json
```
Semantic faithfulness is enabled by default in shadow mode (non-blocking).  
Modes:
- `--semantic-mode offline` (default, local embeddings + contradiction heuristics)
- `--semantic-mode online` (OpenAI-compatible online judge)
- `--semantic-mode hybrid` (offline + online)

Optional controls:
```bash
python -m src.cli validate-report \
  --input runs/research_reports/report.md \
  --evidence runs/research_reports/report.evidence.json \
  --semantic-mode hybrid \
  --semantic-model sentence-transformers/all-MiniLM-L6-v2 \
  --semantic-min-support 0.55 \
  --semantic-max-contradiction 0.30 \
  --semantic-shadow-mode \
  --online-semantic-model gpt-4o-mini \
  --online-semantic-timeout-sec 12 \
  --online-semantic-max-checks 12 \
  --online-semantic-on-warn-only \
  --online-semantic-base-url https://api.openai.com/v1 \
  --online-semantic-api-key "$OPENAI_API_KEY" \
  --semantic-fail-on-low-support
```
8. Benchmark runtime/caching:
```bash
python -m src.cli benchmark-research --index data/indexes/bm25_index.json --queries-file tests/fixtures/queries.txt --runs-per-query 3 --out runs/research_reports/benchmark.json
```
Hybrid benchmark example:
```bash
python -m src.cli benchmark-research --index data/indexes/bm25_index.json --queries-file tests/fixtures/queries.txt --retrieval-mode hybrid --alpha 0.6 --out runs/research_reports/benchmark_hybrid.json
```
Vector benchmark output now includes `vector_eval`:
- `recall_at_k` vs exact flat baseline
- ANN/exact p50/p95 latency
- gate decision for recall non-regression + p95 improvement

Vector service helper commands:
```bash
python -m src.cli vector-service-build --index data/indexes/bm25_index.json --vector-index-type hnsw --vector-m 32 --vector-ef-construction 200
python -m src.cli vector-service-serve --index data/indexes/bm25_index.json --host 127.0.0.1 --port 8765
python -m src.cli vector-service-health --index data/indexes/bm25_index.json
python -m src.cli vector-service-query --index data/indexes/bm25_index.json --question "mobile segmentation limits" --top-k 8
```
Route `research` through the service (with automatic in-process fallback if unreachable):
```bash
python -m src.cli research --index data/indexes/bm25_index.json --question "mobile segmentation limits" --retrieval-mode hybrid --vector-service-endpoint http://127.0.0.1:8765 --out runs/research_reports/report.md
```
9. Benchmark larger synthetic scale:
```bash
python -m src.cli benchmark-scale --corpus data/extracted --queries-file tests/fixtures/queries.txt --repeat-factor 50 --runs-per-query 2 --out runs/research_reports/benchmark_scale.json
```
10. Create reproducibility snapshot bundle:
```bash
python -m src.cli snapshot-run --report runs/research_reports/report.md --index data/indexes/bm25_index.json --config config/train.yaml --config config/sources.yaml --out runs/snapshots
```

## One-Command Workflows
With `.venv` installed:
```bash
make report-local QUERY="mobile segmentation limitations"
```
This runs: build-index -> research -> validate-report on local extracted corpus.

```bash
make report QUERY="mobile segmentation limitations"
```
This runs: sync-papers -> extract-corpus -> report-local.

Single CLI command equivalent (with early gate failures):
```bash
python -m src.cli run-pipeline \
  --query "mobile segmentation limitations" \
  --sources-config config/sources.yaml \
  --with-semantic-scholar
```
By default, pipeline summary JSON is also written to:
`runs/research_reports/pipeline_latest.json`
You can override with `--summary-out <path>`.
Optional active alert dispatch in this manual flow:
```bash
python -m src.cli run-pipeline \
  --query "mobile segmentation limitations" \
  --dispatch-alerts \
  --alerts-config config/automation.yaml
```
Alert dispatch artifact is written to:
`runs/audit/latest_alert_pipeline.json`

Research command outputs companion files:
- `report.evidence.json`
- `report.json`
- `report.metrics.json` (citation/evidence coverage metrics)

## Query Router Modes
The `query` command is now the primary prompt entrypoint. It routes prompts deterministically into one of:
- `ask`: fast deterministic responses (math, unit conversion, glossary/fallback).
  - definition parsing now handles natural phrasing (e.g., “can you explain X in simple terms?”).
- `research`: full evidence-grounded report path (existing behavior).
- `monitor`: create/update a watch and run an immediate baseline automation run.
- `notes`: run research first, then ingest cited key claims into KB and write structured notes.
  - if `## Key Claims` is missing/uncited in report markdown, NOTES auto-normalizes it from cited synthesis lines before KB ingest.

Auto routing order:
1. explicit `--mode` override
2. monitor intent keywords (`monitor|track|notify|alert`)
3. notes intent keywords (`notes|study notes|summarize as notes|store this`)
4. ask patterns (arithmetic, small conversion/definition prompts)
5. fallback to `research`

Examples:
```bash
python -m src.cli query --question "23 + 34 = ?"
python -m src.cli query --question "What is an algorithm?"
python -m src.cli query --question "Monitor NVIDIA stock and alert me"
python -m src.cli query --question "Create study notes on transformer failure modes" --index data/indexes/bm25_index.json
```

Explicit mode override:
```bash
python -m src.cli query --mode research --question "Compare ViT and CNN tradeoffs with citations" --index data/indexes/bm25_index.json
```

Mode subcommands are also available directly:
```bash
python -m src.cli ask --question "45 km/h to m/s"
python -m src.cli monitor --question "Track PIX outage signals"
python -m src.cli notes --question "Create study notes on boundary failure modes" --index data/indexes/bm25_index.json
```

Router observability:
- `query` writes `*.router.json` sidecars with selected mode, reason, signals, override flag, and dispatch timings.
- ASK mode emits a no-citation notice when citation/literature wording is present:
  - `ASK mode does not provide evidence citations; use RESEARCH mode for grounded citations.`

Monitor scheduling:
- `monitor` now attempts recurring cron registration automatically (idempotent by monitor name).
- Disable if needed with:
```bash
python -m src.cli monitor --question "Track PIX outage signals" --no-register-schedule
```
- Remove a registered monitor schedule (and optionally generated files):
```bash
python -m src.cli monitor-unregister --name track_pix_outage_signals --delete-files
```

Monitor soak simulation (multi-day behavior compressed):
```bash
python -m src.cli monitor-soak-sim \
  --topic finance_risk \
  --runs 96 \
  --interval-minutes 60 \
  --cooldown-minutes 360 \
  --hysteresis-runs 2 \
  --pattern constant_bad \
  --out runs/audit/monitor_soak_finance.json
```

Router calibration/eval:
- Run deterministic routing evaluation on a labeled case set:
```bash
python -m src.cli query-router-eval \
  --cases tests/fixtures/router_eval_cases.json \
  --config config/router.yaml \
  --out runs/audit/router_eval.json
```
Optional strict gate (non-zero exit if accuracy drops below threshold):
```bash
python -m src.cli query-router-eval \
  --cases tests/fixtures/router_eval_cases.json \
  --strict-min-accuracy 0.95
```

Retrieval behavior:
- CV-aware query expansion is applied automatically (e.g., segmentation -> matting/mask/foreground).
- Section-aware weighting boosts higher-signal sections (`results`, `limitations`, `future_work`).
- Per-paper diversification limits repeated snippets from the same paper.
- Score blending applies light metadata priors (recency, venue, citation count).
- Shortlist reasons now include metadata signals (year, venue, citation count).
- Index loading uses in-process cache; repeated queries support on-disk report cache.

Sync behavior:
- Source connectors use retry/backoff for transient HTTP failures.
- `sync-papers` reports per-source fetch counts and `source_errors`.
- `sync-papers` also reports PDF failure categories: `missing_pdf_url`, `download_http_error`, `non_pdf_content_type`, `blocked_or_paywalled`.
- Optional policies: `--prefer-arxiv` and `--require-pdf`.
- `sync-papers` now reports source quality metrics and gate status:
  - `source_quality_healthy`
  - `source_quality_reasons`
  - `source_quality_metrics`
- To hard-fail a run when gate thresholds are violated:
```bash
python -m src.cli sync-papers --query "..." --fail-on-source-quality-gate
```

Faithfulness behavior:
- `validate-report` now checks claim-to-evidence lexical support for cited synthesis lines.
- `validate-report` also supports semantic faithfulness scoring in `offline|online|hybrid` modes with shadow/fail rollout flags.

Extraction eval authoring helper:
```bash
python -m src.cli extraction-eval-scaffold-gold \
  --corpus data/extracted \
  --out tests/fixtures/extraction_gold.generated.json \
  --max-papers 20 \
  --checks-per-paper 2
```
Extraction eval supports flexible paper matching:
- primary: `paper_id`
- fallback selectors in gold rows: `doi`, `arxiv_id`, `title`

## Automation Runbook (Cron-Based)
Use the small-to-big automation path with config in `config/automation.yaml`.

1. Dry-run locally:
```bash
make auto-update
```
2. Outputs:
- Per-run artifacts: `runs/audit/runs/<run_id>/manifest.json`
- Latest summaries: `runs/audit/latest_summary.json`, `runs/audit/latest_summary.md`
- Latest alert dispatch status: `runs/audit/latest_alert.json`
- Topic reports: `runs/research_reports/automation/<run_id>/`
- Automation now runs a corpus health gate after extraction and before index build.

3. Optional active alerting (webhook + Gmail SMTP):
Set in `config/automation.yaml`:
```yaml
alerts:
  enabled: true
  webhook_url: "https://hooks.slack.com/services/..."
  webhook_timeout_sec: 10
  webhook_headers:
    Authorization: "Bearer <token>"
  on_corpus_unhealthy: true
  on_topic_validation_failed: true
  on_source_errors: true
  email_enabled: true
  email_to:
    - "comjiji7@gmail.com"
  email_from: "comjiji7@gmail.com"
  smtp_host: "smtp.gmail.com"
  smtp_port: 465
  smtp_use_ssl: true
  smtp_username: "comjiji7@gmail.com"
  smtp_password_env: "GMAIL_APP_PASSWORD"
```
Alerts are evaluated from run summary and dispatched by `scripts/post_run_summary.py`.
Gmail auth: create an App Password and export it before running automation:
```bash
export GMAIL_APP_PASSWORD="xxxx xxxx xxxx xxxx"
```

4. Install weekly cron (Sunday 2AM local):
```bash
crontab -e
```
Add:
```cron
0 2 * * 0 /home/ela/Work-Force/Artifacts/Yipoodle/scripts/auto_update.sh >> /home/ela/Work-Force/Artifacts/Yipoodle/runs/audit/cron.log 2>&1
```

5. Remove cron entry:
```bash
crontab -e
```
Delete the line above.

6. Replay latest summary (+ alert dispatch):
```bash
.venv/bin/python scripts/post_run_summary.py --audit-dir runs/audit --config config/automation.yaml
```

7. Recover/replay a failed topic manually:
```bash
.venv/bin/python -m src.cli research --index data/indexes/automation_index.json --question "<topic query>" --top-k 12 --min-items 4 --min-score 0.15 --out runs/research_reports/replay.md
.venv/bin/python -m src.cli validate-report --input runs/research_reports/replay.md --evidence runs/research_reports/replay.evidence.json
```
