# Architecture

This document describes the system architecture of Yipoodle, covering module responsibilities, data flow, storage design, and integration patterns.

---

## High-Level Architecture

Yipoodle is organized into four major subsystems, all accessed through a single CLI entry point:

```
┌───────────────────────────────────────────────────────────────────────┐
│                           CLI Layer                                   │
│                       src/cli.py (~40 cmds)                           │
└────────┬──────────┬──────────────┬────────────────┬───────────────────┘
         │          │              │                │
    ┌────▼────┐ ┌───▼────┐  ┌─────▼──────┐  ┌─────▼──────────┐
    │Research │ │Tiny GPT│  │Query Router│  │  Automation    │
    │Pipeline │ │ Engine │  │  + Modes   │  │    Engine      │
    └────┬────┘ └───┬────┘  └─────┬──────┘  └─────┬──────────┘
         │          │             │                │
    ┌────▼──────────▼─────────────▼────────────────▼────────────┐
    │                   Core Data Layer                          │
    │   Schemas · Validation · IDs · Semantic Checking           │
    └───────────────────────────┬────────────────────────────────┘
                                │
    ┌───────────────────────────▼────────────────────────────────┐
    │                   Storage Layer                             │
    │   SQLite · JSON · FAISS · Markdown · YAML                  │
    └────────────────────────────────────────────────────────────┘
```

---

## Subsystem 1: Research Pipeline

The research pipeline is the primary subsystem. It automates the full lifecycle of academic research:

### Stage Flow

```
Paper Discovery ─→ PDF Extraction ─→ Corpus Health ─→ Index Build ─→ Retrieval ─→ Synthesis ─→ Validation
     │                   │                │               │              │            │             │
  sync-papers      extract-corpus   corpus-health    build-index    research    (internal)   validate-report
     │                   │                │               │              │            │             │
  paper_search.py  evidence_extract   corpus_health   index_builder  retrieval  research_      validation.py
  paper_sync.py       .py               .py              .py           .py     copilot.py     semantic_*.py
  paper_ingest.py
```

### Paper Discovery (`src/apps/paper_search.py`, `paper_sync.py`, `paper_ingest.py`)

- **18 source connectors**: arXiv, OpenAlex, Semantic Scholar, Crossref, DBLP, PapersWithCode, CORE, OpenReview, GitHub, Zenodo, OpenCitations, Springer, IEEE Xplore, Figshare, OpenML, GDELT, Wikidata, ORCID.
- Each connector returns `list[PaperRecord]` with normalized metadata.
- `sync_papers()` orchestrates: parallel source fetch → deduplication (DOI + title hash) → optional Unpaywall enrichment → PDF download → SQLite upsert.
- Source quality gates evaluate error rates, download success, paywall rates.
- All HTTP calls use `_request_get_with_retry()` with exponential backoff.

### PDF Extraction (`src/apps/evidence_extract.py`, `layout_engine.py`)

- **Extractor fallback chain**: pypdf (always available) → pymupdf (optional) → pdfminer.six (optional).
- **Two-column reconstruction**: detects academic two-column layouts and reorders text.
- **Layout engine modes**: `legacy` (simple extraction), `v2` (region-based with confidence scoring), `shadow` (v2 diagnostics with legacy output), `auto` (uses promotion gate state).
- **OCR fallback**: Tesseract CLI for low-text PDFs with configurable triggers, language detection, noise suppression.
- Output: per-paper JSON files with text snippets, page stats, extraction metadata, and quality scores.

### Corpus Health (`src/apps/corpus_health.py`)

- Pre-indexing quality gate checking: minimum snippet count, average chars per paper, average chars per page, extraction error rate.
- Returns `healthy` boolean with reasons and warnings.
- Used as a blocking gate before index building.

### Index Building (`src/apps/index_builder.py`)

- **BM25 lexical index**: custom implementation with IDF weighting, section-aware term boosting.
- **FAISS vector index** (optional): supports `flat`, `ivf_flat`, `hnsw` backends with configurable sharding via SHA1 hash bucketing.
- Enriches snippet metadata (year, venue, citation count) from SQLite paper DB.
- Can merge live data snippets into the index.

### Retrieval (`src/apps/retrieval.py`, `vector_index.py`)

- **Three retrieval modes**: lexical (BM25), vector (FAISS + sentence-transformers), hybrid (alpha-weighted fusion).
- **Scoring priors**: year recency decay (0.85–1.10), venue tier boost (NeurIPS, ICML, CVPR = 1.08), citation log boost, extraction quality weight.
- **Section-aware weighting**: boosts `results`, `limitations`, `future_work` sections.
- **Per-paper diversification**: limits repeated snippets from the same paper.
- Index loading uses LRU cache with mtime invalidation.

### Research Synthesis (`src/apps/research_copilot.py`)

The heart of the system (~1260 lines). Orchestrates:

1. Cache check (deterministic key from ~40 parameters)
2. Direct answer short-circuit (arithmetic/conversion)
3. Retrieval (lexical/vector/hybrid)
4. Optional live source integration (intent-routed)
5. Optional KB merge (advisory)
6. Relevance gating (configurable policy: not_found/warn/fail)
7. Report building (synthesis, shortlist, claims, gaps, experiments)
8. Semantic validation (shadow mode)
9. Metrics computation (citation coverage, evidence usage)
10. Output writing (markdown + JSON evidence + metrics)
11. Cache storage

**Key design decision**: entirely deterministic — no LLM inference. Synthesis is built from term overlap, TF-IDF scoring, and template rendering. Every sentence traces to a snippet ID.

### Validation (`src/core/validation.py`, `semantic_validation.py`, `semantic_online.py`)

Multi-layer validation pipeline:

1. **Citation format**: regex-based check for `(P<id>:S<n>)` and `(SNAP:<id>:S<n>)` patterns.
2. **Fabrication detection**: flags numbers in synthesis not found in evidence.
3. **Claim support**: token overlap between claims and cited evidence snippets.
4. **Semantic faithfulness** (offline): cosine similarity via sentence-transformers with contradiction proxy (negation detection, direction conflicts, numeric mismatches).
5. **Semantic faithfulness** (online, optional): OpenAI-compatible judge for high-confidence checking.
6. **Shadow mode**: logs semantic results without blocking the pipeline.

---

## Subsystem 2: Tiny GPT

A from-scratch character-level GPT transformer for local text generation.

### Components

| Module                | Responsibility                                                                                                                      |
| --------------------- | ----------------------------------------------------------------------------------------------------------------------------------- |
| `src/model/layers.py` | `CausalSelfAttention` (multi-head + causal mask), `MLP` (4× GELU expansion), `Block` (pre-norm transformer block)                   |
| `src/model/gpt.py`    | `MiniGPT` model — token + position embeddings → N blocks → LayerNorm → linear head. Autoregressive generation with top-k filtering. |
| `src/config.py`       | `TinyGPTConfig` dataclass — batch_size, block_size, n_embd, n_head, n_layer, dropout, lr, temperature, top_k                        |
| `src/tokenizer.py`    | `CharTokenizer` — character-level encode/decode with stoi/itos mappings                                                             |
| `src/dataset.py`      | `TextDataset` — 90/10 train/val split, `get_batch()` returns context windows                                                        |
| `src/train.py`        | Training loop — AdamW optimizer, gradient clipping (max_norm=1.0), checkpoint saving                                                |
| `src/checkpoint.py`   | Save/load with `weights_only=True` and strict payload validation                                                                    |
| `src/generate.py`     | Text generation from checkpoint with temperature and top-k sampling                                                                 |

### Default Hyperparameters

| Parameter            | Value |
| -------------------- | ----- |
| Embedding dim        | 128   |
| Attention heads      | 4     |
| Transformer layers   | 4     |
| Block size (context) | 128   |
| Batch size           | 16    |
| Learning rate        | 3e-4  |
| Dropout              | 0.1   |
| Max training steps   | 1000  |

---

## Subsystem 3: Query Router

Routes user prompts to the appropriate handling mode.

### Routing Pipeline

```
User prompt
    │
    ├─ explicit --mode override? ──────────→ Direct dispatch
    │
    ├─ monitor keywords? ─────────────────→ MONITOR mode
    │  (monitor, track, notify, alert)
    │
    ├─ notes keywords? ───────────────────→ NOTES mode
    │  (notes, study notes, summarize as notes)
    │
    ├─ arithmetic pattern? ───────────────→ ASK mode
    │  (e.g., "23 + 34 = ?")
    │
    ├─ unit conversion / definition? ─────→ ASK mode
    │  (e.g., "45 km/h to m/s", "What is X?")
    │
    └─ fallback ──────────────────────────→ RESEARCH mode
```

### Mode Implementations

| Mode     | Module                | Behavior                                                                                                         |
| -------- | --------------------- | ---------------------------------------------------------------------------------------------------------------- |
| ASK      | `ask_mode.py`         | Deterministic answers: arithmetic eval, unit conversion, glossary lookup, definition fallback. No corpus access. |
| RESEARCH | `research_copilot.py` | Full evidence-grounded report with retrieval, synthesis, validation, metrics.                                    |
| MONITOR  | `monitor_mode.py`     | Creates recurring topic watch, registers cron schedule, runs baseline automation.                                |
| NOTES    | `notes_mode.py`       | Runs research → parses key claims → ingests to KB → generates structured study notes.                            |

---

## Subsystem 4: Automation Engine

Scheduled research automation with alerting.

### Automation Pipeline

```
config/automation.yaml
    │
    ▼
run_automation()
    │
    ├─ Per-topic: sync-papers (subprocess)
    ├─ Optional: layout-promotion-gate
    ├─ Per-topic: extract-corpus (subprocess)
    ├─ Corpus health gate
    ├─ Per-topic: build-index (subprocess)
    ├─ Per-topic: research (subprocess)
    ├─ Per-topic: validate-report (subprocess)
    ├─ Optional: KB ingest + KB diff
    ├─ Optional: monitoring evaluation
    ├─ Optional: benchmark
    ├─ Optional: snapshot bundle
    │
    ▼
Run manifest + summary JSON/MD
    │
    ▼
dispatch_alerts() → Webhook + Gmail SMTP
```

**Design note**: Each stage runs as a separate `python -m src.cli` subprocess for maximal isolation. This adds process startup overhead but prevents cascading failures.

### Alert Transports

- **Webhook**: POST JSON to any URL (Slack, Teams, custom) with configurable headers and timeout.
- **Email**: Gmail SMTP_SSL with App Password authentication.
- **Triggers**: corpus unhealthy, topic validation failed, source fetch errors.

### Scheduling

- Cron-based via `scripts/auto_update.sh` with filesystem locking.
- Per-monitor cron registration (crontab or file-based fallback).
- Cooldown and hysteresis for noise control.

---

## Knowledge Base

Long-lived grounded memory system backed by SQLite.

### Schema (7 tables)

| Table              | Purpose                                                          |
| ------------------ | ---------------------------------------------------------------- |
| `kb_topic`         | Topic namespaces (e.g., "finance_markets")                       |
| `kb_claim`         | Versioned knowledge claims with confidence scores (FTS5 indexed) |
| `kb_claim_version` | Full version history for all claim updates                       |
| `kb_evidence`      | Source evidence linked to claims                                 |
| `kb_contradiction` | Auto-detected contradictions between claims                      |
| `kb_note`          | Structured study notes                                           |
| `kb_change`        | Change tracking for diffs                                        |

### Confidence Model

- Claims start at confidence based on evidence strength.
- Time-based daily decay (configurable rate and stale threshold).
- Contradiction detection via negation word overlap between claims.
- Advisory merge into retrieval results with configurable merge weight.

---

## Live Sources

Real-time data integration layer.

### Source Types

| Type   | Implementation                          | Examples                  |
| ------ | --------------------------------------- | ------------------------- |
| REST   | JSON API fetch with JSONPath extraction | Yahoo Finance, Open-Meteo |
| RSS    | Atom/RSS feed parsing                   | HackerNews, BBC Sport     |
| Scrape | HTML fetch with tag stripping           | Generic web pages         |

### Features

- Rate limiting (requests per minute).
- Auth support (bearer token, custom headers).
- Snapshot caching with configurable TTL.
- Intent-routed source selection (finance keywords → finance sources).
- Live evidence receives `SNAP:<snapshot_id>:S<idx>` citation IDs.

---

## Data Flow Summary

```
External APIs ──→ PaperRecord ──→ SQLite (papers.db)
                                      │
Downloaded PDFs ──→ Extraction ──→ SnippetRecord JSON
                                      │
SnippetRecords ──→ BM25 Index ──→ JSON file
                  ──→ FAISS Index ──→ .faiss + .vector_meta.json
                                      │
Query ──→ Retrieval ──→ EvidenceItem[] ──→ EvidencePack
                                              │
EvidencePack ──→ ResearchReport ──→ Markdown + evidence.json + metrics.json
                                              │
Report ──→ Validation ──→ Pass/Fail + diagnostics
                                              │
Claims ──→ KB Ingest ──→ SQLite (knowledge.db)
```

---

## Key Design Decisions

1. **No LLM for synthesis**: All report text is derived from source snippets via term overlap and template rendering. This guarantees traceability but limits prose quality.

2. **CLI-as-orchestrator**: Every operation is a CLI subcommand. The automation engine calls CLI commands as subprocesses rather than importing functions directly, providing process isolation.

3. **Multi-stage quality gating**: Source quality → corpus health → relevance gate → citation validation → semantic faithfulness. Each gate can block or warn.

4. **Shadow mode rollout**: New validation features (semantic checking, layout v2) deploy in shadow mode first — they log results without blocking — then promote to enforced mode.

5. **Offline-first**: The system works entirely offline after initial paper sync. No internet required for extraction, indexing, retrieval, synthesis, or validation.

6. **Cache everywhere**: LRU caches for index loading and embedding models; file-based caches for research results, Unpaywall lookups, and live snapshots.
