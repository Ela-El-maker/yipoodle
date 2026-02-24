# Folder Structure

Detailed explanation of every directory and key file in the Yipoodle project.

---

## Root

```
yipoodle/
├── Makefile               # Task runner with common workflows
├── pytest.ini             # pytest configuration (sets pythonpath)
├── README.md              # Project overview and quick start
├── requirements.txt       # Development dependencies
```

---

## `config/`

YAML configuration files that control system behavior.

```
config/
├── automation.yaml        # Automation engine: topics, thresholds, alerts, KB, monitoring
├── ask_glossary.yaml      # Term definitions for ASK mode (algorithm, overfitting, etc.)
├── router.yaml            # Query router: intent keywords, patterns, cron defaults
├── sources.yaml           # Default source connectors, retrieval ranking, live sources
├── train.yaml             # Tiny GPT hyperparameters (batch_size, n_embd, n_layer, etc.)
├── domains/               # Domain-specific source configs
│   ├── README.md          # Domain config documentation
│   ├── sources_biomed_ai.yaml
│   ├── sources_cloud_infrastructure.yaml
│   ├── sources_cybersecurity.yaml
│   ├── sources_data_engineering.yaml
│   ├── sources_finance_markets.yaml
│   ├── sources_multimodal_ai.yaml
│   ├── sources_nlp.yaml
│   ├── sources_reinforcement_learning.yaml
│   ├── sources_robotics.yaml
│   ├── sources_software_engineering.yaml
│   └── sources_speech_audio.yaml
└── prompts/               # Reserved for prompt templates (currently empty)
```

---

## `src/`

All source code. Organized into four packages.

```
src/
├── __init__.py            # Package marker
├── cli.py                 # Main CLI entry point (~1600 lines, ~40 subcommands)
├── config.py              # TinyGPTConfig dataclass with YAML/JSON loading
├── train.py               # Training loop (AdamW, gradient clipping, checkpoint saving)
├── generate.py            # Text generation from checkpoint with top-k sampling
├── checkpoint.py          # Secure checkpoint save/load (weights_only=True)
├── tokenizer.py           # CharTokenizer: character-level encode/decode
├── dataset.py             # TextDataset: 90/10 split, get_batch() for training
├── utils.py               # Shared utilities: set_seed, get_device, timestamp
```

### `src/model/`

Transformer architecture (from scratch, no Hugging Face).

```
src/model/
├── __init__.py
├── layers.py              # CausalSelfAttention, MLP (4× GELU), Block (pre-norm)
└── gpt.py                 # MiniGPT: embeddings → blocks → head, with generate()
```

### `src/core/`

Data schemas, validation, and shared types.

```
src/core/
├── __init__.py
├── schemas.py             # Pydantic models: PaperRecord, SnippetRecord, EvidenceItem,
│                          #   EvidencePack, ResearchReport, ShortlistItem, etc.
├── validation.py          # Report validation: citation format, fabrication detection,
│                          #   claim support, coverage metrics, semantic orchestration
├── ids.py                 # ID normalization: normalize_id(), snippet_id()
├── semantic_validation.py # Offline semantic faithfulness: cosine similarity,
│                          #   contradiction proxy (negation, direction, numeric)
├── semantic_online.py     # Online semantic judge (OpenAI-compatible API)
└── live_schema.py         # Live source data models: LiveSourceConfig, LiveSnippet
```

### `src/apps/`

Application modules (~45 files). The main functional logic.

```
src/apps/
├── __init__.py
│
│  ── Research Pipeline ──
├── research_copilot.py    # Research orchestrator (~1260 lines): retrieval → synthesis → validation
├── retrieval.py           # SimpleBM25Index: BM25 scoring, metadata priors, hybrid fusion
├── paper_search.py        # 18 source connectors (arXiv, OpenAlex, Semantic Scholar, etc.)
├── paper_sync.py          # Multi-source fetch, dedup, Unpaywall enrichment, PDF download
├── paper_ingest.py        # SQLite paper DB: init, upsert, download_pdf_with_status
├── evidence_extract.py    # PDF text extraction with fallback chain + two-column + OCR
├── index_builder.py       # BM25 + optional FAISS index building from corpus
├── pipeline_runner.py     # One-command pipeline: sync → extract → health → index → research → validate
├── research_report.py     # Report rendering helpers
│
│  ── Query Modes ──
├── query_router.py        # Rule-based prompt routing: ask/research/monitor/notes
├── query_router_eval.py   # Router accuracy evaluation harness
├── ask_mode.py            # Deterministic answers: arithmetic, conversion, glossary
├── direct_answer.py       # Direct arithmetic evaluation
├── monitor_mode.py        # Monitor topic creation, cron scheduling, baseline run
├── notes_mode.py          # Research + KB ingest + structured study notes
├── query_builder.py       # Domain-aware query expansion
│
│  ── Knowledge Base ──
├── kb_store.py            # SQLite KB: 7 tables, FTS5, versioning, confidence decay
├── kb_ingest.py           # Ingest report claims into KB
├── kb_query.py            # Query KB claims with topic filter
├── kb_diff.py             # KB change tracking (diff since run/timestamp)
├── kb_claim_parser.py     # Parse "## Key Claims" from report markdown
├── kb_confidence.py       # Confidence decay and scoring
│
│  ── Monitoring ──
├── monitoring_engine.py   # Monitoring rule evaluation
├── monitoring_rules.py    # Rule definitions (breakthrough, citation velocity, trending)
├── monitoring_state.py    # Persistent monitoring state
├── monitoring_diff.py     # Monitoring diff detection
├── monitoring_history.py  # Trigger history analysis
├── monitoring_hooks.py    # On-trigger/on-clear hook dispatch
├── monitoring_soak.py     # Multi-day soak simulation
│
│  ── Vector Search ──
├── vector_index.py        # FAISS index: build, save, load, query, sharding
├── vector_backend.py      # FAISS backend factory (flat, ivf_flat, hnsw)
├── vector_service.py      # Vector HTTP microservice (ThreadingHTTPServer)
│
│  ── Live Sources ──
├── live_sources.py        # Live data fetchers: REST, RSS, scrape
├── live_routing.py        # Intent-based source routing
├── live_snapshot_store.py  # Live snapshot persistence
│
│  ── Corpus & Quality ──
├── corpus_health.py       # Corpus quality gate evaluation
├── corpus_migration.py    # Legacy corpus metadata migration
├── extraction_eval.py     # Extraction fidelity evaluation (gold checks)
├── extraction_quality_report.py  # Per-PDF quality reporting
├── layout_engine.py       # Layout v2 engine (region-based extraction)
├── layout_promotion.py    # Layout v2 promotion gate (shadow → enforced)
│
│  ── Automation ──
├── automation.py          # Cron automation engine (~1180 lines): run orchestration, alerts
├── sources_config.py      # Sources YAML config loader
│
│  ── Utility ──
├── doc_writer.py          # Documentation generation from structured facts
├── release_notes.py       # Release notes generation between git refs
├── domain_scaffold.py     # Domain config template generation
├── snapshot.py            # Reproducibility snapshot bundling
└── benchmark.py           # Benchmark utilities
```

---

## `data/`

Runtime data directory. Contents are generated by the pipeline.

```
data/
├── data.txt               # Training text corpus for Tiny GPT
├── automation/            # Automation-specific data
├── extracted/             # Extracted paper text (per-paper JSON files)
├── extracted_cv/          # CV-domain extracted corpus
├── extracted_cv_db/       # CV-domain with DB enrichment
├── indexes/               # BM25 JSON indexes + FAISS sidecar files
├── kb/                    # Knowledge base SQLite databases
├── live_snapshots/        # Cached live source snapshots
├── papers/                # Downloaded PDF files
├── papers_cv/             # CV-domain papers
├── reports/               # Generated data reports
└── training/              # Training data artifacts
```

---

## `runs/`

Runtime outputs. Generated during pipeline execution.

```
runs/
├── audit/                 # Automation audit trails, run manifests, summaries
│   └── runs/              # Per-run directories with manifest.json
├── cache/                 # Query result cache (research, unpaywall)
├── checkpoints/           # Tiny GPT model checkpoints (.pt files)
├── live/                  # Live source fetch results
├── logs/                  # Training loss logs
├── notes/                 # Generated study notes
├── research_reports/      # Research reports, evidence, metrics
│   └── automation/        # Per-automation-run topic reports
├── smoke/                 # Smoke test artifacts
└── snapshots/             # Reproducibility snapshot bundles
```

---

## `deploy/`

Production deployment profiles.

```
deploy/
├── requirements.base.txt  # Core dependencies (no PyTorch)
├── requirements.cpu.txt   # CPU profile (PyTorch + FAISS CPU)
└── requirements.cuda12.txt # CUDA 12.1 profile
```

---

## `docker/`

Container build files.

```
docker/
├── Dockerfile.cpu         # Python 3.11-slim + CPU dependencies
└── Dockerfile.cuda        # PyTorch CUDA 12.1 base image
```

---

## `scripts/`

Automation and utility scripts.

```
scripts/
├── auto_update.sh         # Cron entry point with filesystem locking
├── auto_update.py         # Python wrapper for run_automation()
└── post_run_summary.py    # Generate summary + dispatch alerts after automation run
```

---

## `tests/`

Test suite (254 tests).

```
tests/
├── fixtures/              # Test fixtures: gold checks, queries, router cases
├── test_apps.py           # Application integration tests
├── test_ask_mode.py       # ASK mode tests
├── test_automation*.py    # Automation engine tests
├── test_benchmark*.py     # Benchmark tests
├── test_cli_*.py          # CLI dispatch tests
├── test_corpus_*.py       # Corpus health and migration tests
├── test_dataset.py        # Dataset tests
├── test_extraction_*.py   # Extraction tests
├── test_faithfulness.py   # Faithfulness validation tests
├── test_hybrid_*.py       # Hybrid retrieval tests
├── test_kb_*.py           # Knowledge base tests
├── test_live_*.py         # Live sources tests
├── test_monitor_*.py      # Monitoring tests
├── test_notes_mode.py     # Notes mode tests
├── test_paper_*.py        # Paper search/sync tests
├── test_pipeline_*.py     # Pipeline runner tests
├── test_query_*.py        # Query router tests
├── test_research_*.py     # Research copilot tests
├── test_retrieval_*.py    # Retrieval scoring tests
├── test_save_load.py      # Checkpoint tests
├── test_schemas.py        # Schema validation tests
├── test_semantic_*.py     # Semantic validation tests
├── test_shapes.py         # Model shape tests
├── test_tokenizer.py      # Tokenizer tests
├── test_vector_*.py       # Vector index/service tests
└── test_validators.py     # Validation tests
```

---

## `docs/`

Project documentation.

```
docs/
├── architecture.md        # System design and module relationships
├── cli-reference.md       # All CLI commands with arguments and examples
├── configuration.md       # YAML config files and tuning options
├── contributing.md        # Development workflow and PR guidelines
├── deployment.md          # Docker, production profiles, cron setup
├── folder-structure.md    # This file
├── known-issues.md        # Limitations and improvement opportunities
├── system-overview.md     # Purpose, philosophy, key concepts
└── testing.md             # Test suite structure and writing tests
```
