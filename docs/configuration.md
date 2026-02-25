# Configuration Guide

This document describes all YAML configuration files in Yipoodle, their structure, and tuning options.

---

## Configuration Files

| File                       | Purpose                                                      |
| -------------------------- | ------------------------------------------------------------ |
| `config/train.yaml`        | Tiny GPT model hyperparameters                               |
| `config/sources.yaml`      | Default source connectors, retrieval, and live source config |
| `config/automation.yaml`   | Automation engine, alerting, topics, and monitoring          |
| `config/router.yaml`       | Query router rules and thresholds                            |
| `config/ask_glossary.yaml` | ASK mode glossary term definitions                           |
| `config/templates.yaml`    | Research session templates (multi-query workflows)           |
| `config/domains/*.yaml`    | Domain-specific source configs (NLP, robotics, etc.)         |

---

## config/train.yaml

Tiny GPT model hyperparameters. Maps directly to `TinyGPTConfig` fields.

```yaml
batch_size: 16
block_size: 128
n_embd: 128
n_head: 4
n_layer: 4
dropout: 0.1
lr: 0.0003
max_steps: 1000
eval_every: 100
gen_tokens: 256
temperature: 0.9
top_k: 40
seed: 1337
```

| Parameter     | Type  | Default | Description                        |
| ------------- | ----- | ------- | ---------------------------------- |
| `batch_size`  | int   | 16      | Training batch size                |
| `block_size`  | int   | 128     | Context window length (characters) |
| `n_embd`      | int   | 128     | Embedding dimension                |
| `n_head`      | int   | 4       | Number of attention heads          |
| `n_layer`     | int   | 4       | Number of transformer blocks       |
| `dropout`     | float | 0.1     | Dropout rate                       |
| `lr`          | float | 3e-4    | Learning rate (AdamW)              |
| `max_steps`   | int   | 1000    | Maximum training steps             |
| `eval_every`  | int   | 100     | Evaluate every N steps             |
| `gen_tokens`  | int   | 256     | Tokens to generate in eval         |
| `temperature` | float | 0.9     | Sampling temperature               |
| `top_k`       | int   | 40      | Top-k filtering for generation     |
| `seed`        | int   | 1337    | Random seed                        |

---

## config/sources.yaml

The main configuration for paper sources, retrieval behavior, and live data.

### Top-Level Sections

#### `domain`

```yaml
domain: computer_vision
```

Domain tag used for query expansion and source routing.

#### `ocr`

OCR defaults (overridable by CLI flags):

```yaml
ocr:
  enabled: false
  min_chars_trigger: 120
  timeout_sec: 30
  max_pages: 20
  min_output_chars: 200
  min_gain_chars: 40
  min_confidence: 45.0
  lang: eng
  profile: document
  noise_suppression: true
```

#### `sources`

18 academic API source configurations:

```yaml
sources:
  arxiv:
    enabled: true
    endpoint: "http://export.arxiv.org/api/query"
    max_results: 20
  openalex:
    enabled: true
    endpoint: "https://api.openalex.org/works"
    max_results: 20
  semanticscholar:
    enabled: true
    endpoint: "https://api.semanticscholar.org/graph/v1/paper/search"
    max_results: 20
  crossref:
    enabled: false
    endpoint: "https://api.crossref.org/works"
    max_results: 10
  # ... 14 more sources
```

Each source supports:

- `enabled`: boolean toggle
- `endpoint`: API URL
- `max_results`: per-source result limit
- `auth` (optional): `{"type": "bearer", "token_env": "ENV_VAR"}` or header-based

#### `limits`

```yaml
limits:
  max_total_results: 100
  max_pdf_downloads: 30
  max_tokens_per_summary: 800
  source_quality:
    max_source_error_rate: 0.5
    min_download_success_rate: 0.3
    max_download_http_error_rate: 0.4
    max_non_pdf_content_rate: 0.3
    max_blocked_or_paywalled_rate: 0.5
    max_unpaywall_lookup_error_rate: 0.5
```

#### `ranking`

```yaml
ranking:
  strategy: hybrid
  weights:
    recency: 0.2
    citation_count: 0.15
    semantic_similarity: 0.35
    source_trust: 0.1
```

#### `live_sources`

Live data connectors (REST, RSS, scrape):

```yaml
live_sources:
  yahoo_finance:
    enabled: false
    type: rest
    endpoint: "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
    params:
      interval: "1d"
      range: "5d"
    auth:
      type: bearer
      token_env: YAHOO_FINANCE_API_KEY
    rate_limit_rpm: 5
    cache_ttl_sec: 1800
    domain_tags: [finance, markets, stocks]
```

#### `live_routing`

Intent-based source routing:

```yaml
live_routing:
  finance:
    keywords: [stock, share, market, price, ticker, portfolio, earnings]
    sources: [yahoo_finance]
  tech:
    keywords: [hacker, startup, tech, ycombinator, HN, Show HN]
    sources: [hackernews_rss]
```

#### `live_snapshot`

```yaml
live_snapshot:
  store_path: "data/live_snapshots"
  max_snapshots_per_source: 100
```

---

## config/automation.yaml

Full automation engine configuration.

### `global`

Feature flags applied to all automation runs:

```yaml
global:
  with_semantic_scholar: true
  prefer_arxiv: true
  require_pdf: false
  retrieval_mode: hybrid
  alpha: 0.6
  embedding_model: "sentence-transformers/all-MiniLM-L6-v2"
  vector_index_type: ivf_flat
  vector_nlist: 256
  ocr_enabled: false
  layout_engine: shadow
```

### `paths`

Directory paths for all runtime artifacts:

```yaml
paths:
  db: "data/papers.db"
  papers_dir: "data/papers"
  extracted_dir: "data/extracted"
  index: "data/indexes/automation_index.json"
  reports_dir: "runs/research_reports/automation"
  audit_dir: "runs/audit"
  snapshots_dir: "runs/snapshots"
  cache_dir: "runs/cache"
```

### `thresholds`

Quality gate thresholds:

```yaml
thresholds:
  min_citation_coverage: 0.6
  min_evidence_usage: 0.3
  semantic:
    mode: "offline"
    shadow_mode: true
    min_support: 0.55
    max_contradiction: 0.30
    fail_on_low_support: false
  corpus_health:
    min_snippets: 1
    min_avg_chars_per_paper: 500
    min_avg_chars_per_page: 80
    max_extract_error_rate: 0.8
```

### `alerts`

Alert dispatch configuration:

```yaml
alerts:
  enabled: false
  webhook_url: ""
  webhook_timeout_sec: 10
  webhook_headers: {}
  on_corpus_unhealthy: true
  on_topic_validation_failed: true
  on_source_errors: true
  email_enabled: false
  email_to: []
  email_from: ""
  smtp_host: "smtp.gmail.com"
  smtp_port: 465
  smtp_use_ssl: true
  smtp_username: ""
  smtp_password_env: "GMAIL_APP_PASSWORD"
```

### `kb`

Knowledge base integration:

```yaml
kb:
  enabled: false
  db_path: "data/kb/knowledge.db"
  topic_auto: true
  top_k: 5
  merge_weight: 0.15
  contradiction_resolver_enabled_default: false
  contradiction_resolver_max_pairs: 5
  contradiction_resolver_support_margin: 0.05
```

### `monitoring`

Monitoring behavior:

```yaml
monitoring:
  enabled: false
  baseline_runs: 1
  failure_policy: warn
  noise:
    cooldown_minutes: 360
    hysteresis_runs: 2
  hooks:
    on_trigger: []
    on_clear: []
  digest:
    enabled: true
    interval: daily
    max_events: 50
```

### `reliability`

Connector reliability watchdog (post-sync feedback loop):

```yaml
reliability:
  enabled: false
  db_path: "data/reliability/source_reliability.db"
  state_path: "runs/audit/source_reliability_state.json"
  report_path: "runs/audit/source_reliability.json"
  degrade_threshold: 0.30
  critical_threshold: 0.15
  auto_disable_after: 0
```

### `benchmark_regression`

Benchmark drift gate against historical baseline:

```yaml
benchmark_regression:
  enabled: false
  history_path: "runs/audit/benchmark_history.json"
  max_latency_regression_pct: 10.0
  min_quality_floor: 0.0
  history_window: 104
  run_every_n_runs: 1
```

`run_every_n_runs` lets you gate expensive benchmark checks (for example `5` means every fifth automation run).

### `topics`

Research topics to run:

```yaml
topics:
  cv_mobile_segmentation_limits:
    query: "mobile real-time semantic segmentation edge deployment limitations"
    max_results: 20
    top_k: 12
    min_items: 4
    min_score: 0.15
    monitoring:
      triggers:
        - type: breakthrough
          keywords: [breakthrough, sota, state-of-the-art]
          threshold: 0.7
```

Each topic supports:

- `query`: research question
- `max_results`, `top_k`, `min_items`, `min_score`: retrieval tuning
- `sources_config`: optional domain-specific source config
- `monitoring.triggers`: list of trigger rules (breakthrough, citation_velocity, trending)

---

## config/router.yaml

Query router configuration:

```yaml
ask:
  max_words: 14
  citation_keywords:
    [cite, citation, reference, evidence, literature, sources, studies]
  definition_patterns: ["what is", "define", "explain", "meaning of"]

monitor:
  intent_keywords: [monitor, track, notify, alert]
  default_cron: "0 */6 * * *"

notes:
  intent_keywords: [notes, study notes, summarize as notes, store this]
```

| Setting                   | Description                                               |
| ------------------------- | --------------------------------------------------------- |
| `ask.max_words`           | Maximum words for ASK mode eligibility                    |
| `ask.citation_keywords`   | Keywords that trigger a "use RESEARCH mode" notice in ASK |
| `ask.definition_patterns` | Patterns that route to ASK for definition lookup          |
| `monitor.intent_keywords` | Keywords that trigger MONITOR mode                        |
| `monitor.default_cron`    | Default cron expression for monitor scheduling            |
| `notes.intent_keywords`   | Keywords that trigger NOTES mode                          |

---

## config/ask_glossary.yaml

Glossary terms for ASK mode definition lookups:

```yaml
algorithm: "A step-by-step procedure for solving a problem or performing a computation."
overfitting: "When a model learns noise in the training data instead of the underlying pattern."
amortization: "Spreading the cost of an asset over its useful life in accounting."
```

Add new terms by adding key-value pairs. The ASK mode checks this glossary before falling back to a generic definition response.

---

## config/templates.yaml

Template definitions for `research-template` automation workflow.

```yaml
templates:
  lit_review:
    description: "Literature review sequence"
    questions:
      - "Define the scope in {topic}."
      - "What methods are used in {topic}?"
      - "What evidence gaps remain in {topic}?"
```

Each template question is rendered with `{topic}` substitution and executed through the normal `research` pipeline.

---

## Domain Configs

Domain-specific source configurations under `config/domains/`. Each file overrides `sources.yaml` with domain-appropriate settings.

Available domains:

- `sources_nlp.yaml`
- `sources_biomed_ai.yaml`
- `sources_robotics.yaml`
- `sources_cybersecurity.yaml`
- `sources_multimodal_ai.yaml`
- `sources_speech_audio.yaml`
- `sources_reinforcement_learning.yaml`
- `sources_data_engineering.yaml`
- `sources_cloud_infrastructure.yaml`
- `sources_software_engineering.yaml`
- `sources_finance_markets.yaml`

Generate a new domain config:

```bash
python -m src.cli scaffold-domain-config --domain my_domain --out config/domains/sources_my_domain.yaml
```

---

## Configuration Precedence

For settings that can be specified in multiple places:

```
CLI flags  >  sources-config YAML  >  built-in defaults
```

Example: OCR settings can be set via CLI flags (`--ocr-enabled`), in `sources.yaml` (`ocr:` block), or fall back to hardcoded defaults.
